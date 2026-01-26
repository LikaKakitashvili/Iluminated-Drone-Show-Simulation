import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# =========================
# Utils: reproducibility
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int) -> None:
    np.random.seed(seed)

def normalize_pts01_tight(pts01: np.ndarray, margin: float = 0.06) -> np.ndarray:
    """
    Re-normalize points to [0,1] using their tight bounding box (plus margin).
    This removes empty image margins so the shape fills the world box better.
    """
    mins = pts01.min(axis=0)
    maxs = pts01.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    mins = mins - margin * span
    maxs = maxs + margin * span
    span2 = np.maximum(maxs - mins, 1e-9)
    out = (pts01 - mins) / span2
    return np.clip(out, 0.0, 1.0)

def normalize_pts01_tight_aspect(pts01: np.ndarray, margin: float = 0.08) -> np.ndarray:
    """
    Tight normalization that PRESERVES aspect ratio (single scale for x/y).
    This avoids distorting text into "blocky" shapes.
    """
    mins = pts01.min(axis=0)
    maxs = pts01.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-9)
    s = float(np.max(span))
    out = (pts01 - center) / s + 0.5
    # apply margin by shrinking towards center
    m = float(np.clip(margin, 0.0, 0.49))
    if m > 0:
        out = (out - 0.5) / (1.0 - 2.0 * m) + 0.5
    return np.clip(out, 0.0, 1.0)


# =========================
# Target generation
# =========================

def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Greedy farthest-point sampling for more uniform point distribution.
    This preserves structure better than random sampling.
    """
    if points.shape[0] <= n_samples:
        return points
    
    # Limit candidate size for speed
    if points.shape[0] > 20000:
        idx = np.random.choice(points.shape[0], size=20000, replace=False)
        points = points[idx]
    
    first = np.random.randint(0, points.shape[0])
    selected = [points[first]]
    dist2 = np.full(points.shape[0], np.inf, dtype=np.float64)
    for _ in range(1, n_samples):
        diff = points - selected[-1]
        d2 = np.sum(diff * diff, axis=1)
        dist2 = np.minimum(dist2, d2)
        nxt = int(np.argmax(dist2))
        selected.append(points[nxt])
    return np.array(selected, dtype=np.float64)

def sample_points_from_contours(mask: np.ndarray, n_points: int) -> np.ndarray:
    """
    Sample points uniformly along contour arc-length. This preserves letter structure
    much better than random edge sampling for handwritten words.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("No contours found.")
    
    contours = [c.reshape(-1, 2).astype(np.float64) for c in cnts if c.shape[0] >= 5]
    if not contours:
        raise ValueError("Contours too small.")
    
    # Calculate arc lengths
    lengths = []
    for pts in contours:
        pts_closed = np.vstack([pts, pts[0]])
        seg = np.diff(pts_closed, axis=0)
        L = np.sqrt((seg**2).sum(axis=1)).sum()
        lengths.append(max(L, 1e-9))
    lengths = np.array(lengths, dtype=np.float64)
    total = lengths.sum()
    
    # Allocate samples proportional to contour length
    alloc = np.maximum(1, np.floor(n_points * (lengths / total)).astype(int))
    # Fix rounding
    while alloc.sum() < n_points:
        alloc[np.argmax(lengths)] += 1
    while alloc.sum() > n_points:
        i = int(np.argmax(alloc))
        if alloc[i] > 1:
            alloc[i] -= 1
        else:
            break
    
    sampled = []
    for pts, k in zip(contours, alloc):
        pts_closed = np.vstack([pts, pts[0]])
        seg = np.diff(pts_closed, axis=0)
        seg_len = np.sqrt((seg**2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        L = cum[-1]
        if L <= 0:
            continue
        s = np.linspace(0, L, k, endpoint=False)
        idx = np.searchsorted(cum, s, side="right") - 1
        idx = np.clip(idx, 0, len(seg_len) - 1)
        t = (s - cum[idx]) / (seg_len[idx] + 1e-12)
        p = pts_closed[idx] * (1 - t[:, None]) + pts_closed[idx + 1] * t[:, None]
        sampled.append(p)
    pts = np.vstack(sampled)
    
    # Trim/pad if needed
    if pts.shape[0] > n_points:
        pts = pts[np.random.choice(pts.shape[0], size=n_points, replace=False)]
    elif pts.shape[0] < n_points:
        needed = n_points - pts.shape[0]
        extra = pts[np.random.choice(pts.shape[0], size=needed, replace=True)].copy()
        extra += np.random.normal(scale=0.5, size=extra.shape)
        pts = np.vstack([pts, extra])
    
    h, w = mask.shape
    pts[:, 0] /= (w - 1)
    pts[:, 1] /= (h - 1)
    return np.clip(pts, 0.0, 1.0)

def sample_points_from_binary_mask(mask: np.ndarray, n_points: int, mode: str = "edge") -> np.ndarray:
    """
    mask: uint8 {0,255} with foreground=255
    mode:
      - "edge": sample points from edges using farthest-point sampling
      - "fill": sample points from filled region using farthest-point sampling
    returns: (n_points, 2) points in [0,1]x[0,1] image coordinates (x,y)
    """
    assert mask.ndim == 2
    if mode == "edge":
        edges = cv2.Canny(mask, 50, 150)
        ys, xs = np.where(edges > 0)
    else:
        ys, xs = np.where(mask > 0)

    if len(xs) < max(50, n_points // 4):
        # fallback: use filled if edges are too sparse
        ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise ValueError("No foreground points found in mask. Check your image thresholding.")

    pts_all = np.stack([xs, ys], axis=1).astype(np.float64)
    
    # Use farthest-point sampling for better distribution
    k = min(n_points, pts_all.shape[0])
    pts = farthest_point_sampling(pts_all, k)
    
    # Pad if needed
    if pts.shape[0] < n_points:
        needed = n_points - pts.shape[0]
        extra_idx = np.random.choice(pts.shape[0], size=needed, replace=True)
        extra = pts[extra_idx].copy()
        extra += np.random.normal(scale=0.8, size=extra.shape)
        pts = np.vstack([pts, extra])

    # normalize to [0,1]
    h, w = mask.shape
    pts[:, 0] /= (w - 1)
    pts[:, 1] /= (h - 1)
    return np.clip(pts, 0.0, 1.0)

def load_handwritten_word_points(image_path: str, n_points: int) -> np.ndarray:
    """
    Loads a handwritten word image, binarizes it, and samples target points.
    Uses contour-based sampling for better letter structure preservation.
    For text images, ensures better distribution to prevent edge clustering.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Denoise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Try adaptive threshold first, fallback to Otsu if needed
    try:
        mask = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
    except:
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean noise and ensure connectivity
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Moderate dilation to ensure thin strokes (like 'e') are thick enough for sampling
    # This helps capture letters with thinner strokes better, especially internal counters
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Additional light dilation specifically for thin internal structures (like 'e' counters)
    # This helps ensure small enclosed areas get sampled
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Hybrid sampling: prioritize contour to capture thin strokes and edges
    # Higher contour percentage helps capture 'e' shapes better
    n_contour = int(n_points * 0.70)  # Even more contour to better capture thin strokes like 'e'
    n_fill = n_points - n_contour
    
    try:
        pts_contour = sample_points_from_contours(mask, n_points=n_contour)
    except Exception:
        pts_contour = sample_points_from_binary_mask(mask, n_points=n_contour, mode="edge")
    
    pts_fill = sample_points_from_binary_mask(mask, n_points=n_fill, mode="fill")
    
    # Combine points - preserve natural spacing from image
    pts01 = np.vstack([pts_contour, pts_fill])
    
    # Don't redistribute - preserve the natural spacing from the image
    # The image already has good spacing, so we want to keep that structure
    
    # Normalize with moderate margin to preserve spacing
    return normalize_pts01_tight(pts01, margin=0.10)

def render_text_to_points(text: str, n_points: int, img_size=(1400, 400)) -> np.ndarray:
    """
    Renders text to an image (PIL), converts to mask, samples points.
    Uses hybrid sampling: contour (edges) + fill (body) for better letter definition.
    """
    W, H = img_size
    img = Image.new("L", (W, H), color=255)
    draw = ImageDraw.Draw(img)

    # Try a bold font for thicker strokes
    base_font = None
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        if os.path.exists(candidate):
            base_font = candidate
            break
    
    # Dynamic font sizing to fit text
    font_size = 180
    font = None
    if base_font:
        while font_size >= 24:
            try:
                font = ImageFont.truetype(base_font, font_size)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if tw <= W * 0.92 and th <= H * 0.85:
                    break
                font_size = int(font_size * 0.9)
            except Exception:
                font_size = int(font_size * 0.9)
    if font is None:
        font = ImageFont.load_default()

    # center text with generous horizontal padding to prevent edge compression
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Add horizontal padding (15% on each side) so H and R/! aren't at absolute edges
    x_pad = int(W * 0.15)
    x = x_pad + (W - 2*x_pad - tw) // 2
    y = (H - th) // 2
    draw.text((x, y), text, fill=0, font=font)

    arr = np.array(img)
    # binary mask: letters -> foreground
    mask = (arr < 200).astype(np.uint8) * 255
    
    # Slight dilation to thicken strokes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Hybrid sampling: 50% contour (sharp edges), 50% fill (body)
    # More fill helps with letter body definition, especially for H, R, !
    n_contour = int(n_points * 0.5)
    n_fill = n_points - n_contour
    
    try:
        pts_contour = sample_points_from_contours(mask, n_points=n_contour)
    except Exception:
        pts_contour = sample_points_from_binary_mask(mask, n_points=n_contour, mode="edge")
    
    pts_fill = sample_points_from_binary_mask(mask, n_points=n_fill, mode="fill")
    
    # Combine and shuffle
    pts01 = np.vstack([pts_contour, pts_fill])
    np.random.shuffle(pts01)
    
    # Normalize preserving aspect ratio with larger margin to prevent edge compression
    return normalize_pts01_tight_aspect(pts01, margin=0.12)


# =========================
# Geometry scaling / mapping
# =========================

def pts01_to_world(pts01: np.ndarray, world_box: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Map normalized image coords [0,1] to world coords in a box.
    world_box = (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = world_box
    x = xmin + pts01[:, 0] * (xmax - xmin)
    y = ymax - pts01[:, 1] * (ymax - ymin)  # invert y so image top maps to ymax
    return np.stack([x, y], axis=1)

def init_positions_grid(N: int, box: Tuple[float, float, float, float], jitter: float = 0.02) -> np.ndarray:
    xmin, xmax, ymin, ymax = box
    side = int(math.ceil(math.sqrt(N)))
    xs = np.linspace(xmin, xmax, side)
    ys = np.linspace(ymin, ymax, side)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    grid = grid[:N]
    grid += np.random.uniform(-jitter, jitter, size=grid.shape)
    return grid

def assign_targets_nearest(x: np.ndarray, T: np.ndarray, preserve_order: bool = False) -> np.ndarray:
    """
    Simple assignment: each drone takes a unique target point greedily by nearest.
    If preserve_order=True: uses proportional assignment to prevent edge clustering.
    """
    N = x.shape[0]
    M = T.shape[0]
    assert M >= N

    if preserve_order:
        # Simple proportional assignment: leftmost N/3 drones -> leftmost targets,
        # middle N/3 -> middle targets, rightmost N/3 -> rightmost targets
        # This prevents all left drones from going to H, all right to R/!
        order_x = np.argsort(x[:, 0])
        order_T = np.argsort(T[:, 0])
        
        # Divide into 3 roughly equal groups
        n1 = N // 3
        n2 = (N - n1) // 2
        n3 = N - n1 - n2
        
        # Divide targets into 3 groups
        m1 = M // 3
        m2 = (M - m1) // 2
        m3 = M - m1 - m2
        
        assigned = np.zeros((N, 2), dtype=np.float64)
        
        # Left group: first n1 drones -> first m1 targets
        for i in range(n1):
            di = order_x[i]
            ti_idx = int(i * m1 / max(n1, 1)) % m1
            ti = order_T[ti_idx]
            assigned[di] = T[ti]
        
        # Middle group: middle n2 drones -> middle m2 targets
        for i in range(n2):
            di = order_x[n1 + i]
            ti_idx = m1 + int(i * m2 / max(n2, 1)) % m2
            ti = order_T[ti_idx]
            assigned[di] = T[ti]
        
        # Right group: last n3 drones -> last m3 targets
        for i in range(n3):
            di = order_x[n1 + n2 + i]
            ti_idx = m1 + m2 + int(i * m3 / max(n3, 1)) % m3
            ti = order_T[ti_idx]
            assigned[di] = T[ti]
        
        return assigned
    
    # Original greedy nearest assignment
    remaining = set(range(M))
    assigned = np.zeros((N, 2), dtype=np.float64)

    for i in range(N):
        # find nearest among remaining
        idxs = np.array(list(remaining), dtype=np.int64)
        d2 = np.sum((T[idxs] - x[i])**2, axis=1)
        j = idxs[int(np.argmin(d2))]
        assigned[i] = T[j]
        remaining.remove(int(j))
    return assigned


# =========================
# Swarm dynamics (IVP)
# =========================

@dataclass
class SwarmParams:
    m: float = 1.0
    kp: float = 8.0
    kd: float = 4.0
    krep: float = 0.04
    Rsafe: float = 0.06
    vmax: float = 0.8

def saturate(v: np.ndarray, vmax: float) -> np.ndarray:
    """
    v: (N,2)
    """
    speed = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    factor = np.minimum(1.0, vmax / speed)
    return v * factor

def repulsion_forces(x: np.ndarray, krep: float, Rsafe: float) -> np.ndarray:
    """
    Pairwise repulsion for collision-free behavior:
      frep(xi,xj) = krep*(xi-xj)/||xi-xj||^3 if ||xi-xj|| < Rsafe else 0
    This ensures drones maintain safe distance and prevents collisions.
    returns: (N,2)
    """
    N = x.shape[0]
    F = np.zeros_like(x)
    for i in range(N):
        diff = x[i] - x  # (N,2)
        dist = np.linalg.norm(diff, axis=1) + 1e-12
        # Repulsion active when within Rsafe distance (collision avoidance zone)
        mask = (dist < Rsafe) & (dist > 0)
        if np.any(mask):
            # Stronger repulsion for closer drones (1/dist^3) ensures collision-free
            contrib = krep * (diff[mask] / (dist[mask][:, None]**3))
            F[i] += np.sum(contrib, axis=0)
    return F

def accel_position_tracking(x: np.ndarray, v: np.ndarray, T: np.ndarray, p: SwarmParams) -> np.ndarray:
    """
    vdot = ( kp*(T-x) + sum frep - kd*v )/m
    """
    Fp = p.kp * (T - x)
    Frep = repulsion_forces(x, p.krep, p.Rsafe)
    Fd = -p.kd * v
    a = (Fp + Frep + Fd) / p.m
    return a

def rk4_step(x: np.ndarray, v: np.ndarray, dt: float, T: np.ndarray, p: SwarmParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    xdot = sat(v)
    vdot = a(x,v)
    """
    def f(state_x, state_v):
        vx = saturate(state_v, p.vmax)
        ax = accel_position_tracking(state_x, state_v, T, p)
        return vx, ax

    k1x, k1v = f(x, v)
    k2x, k2v = f(x + 0.5*dt*k1x, v + 0.5*dt*k1v)
    k3x, k3v = f(x + 0.5*dt*k2x, v + 0.5*dt*k2v)
    k4x, k4v = f(x + dt*k3x, v + dt*k3v)

    x_next = x + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
    v_next = v + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return x_next, v_next

def simulate_to_targets(x0: np.ndarray, v0: np.ndarray, T: np.ndarray, p: SwarmParams,
                        dt: float, steps: int) -> np.ndarray:
    """
    returns trajectory array: (steps+1, N, 2)
    """
    x = x0.copy()
    v = v0.copy()
    traj = np.zeros((steps + 1, x.shape[0], 2), dtype=np.float64)
    traj[0] = x
    for k in range(steps):
        x, v = rk4_step(x, v, dt, T, p)
        traj[k+1] = x
    return traj, x, v


# =========================
# IVP with Velocity Tracking (Sub-problem 3)
# =========================

@dataclass
class VTParams:
    m: float = 1.0
    kv: float = 10.0
    kd: float = 3.0
    krep: float = 0.008  # Repulsion for collision-free behavior during velocity tracking
    Rsafe: float = 0.04  # Safety distance - drones within this distance repel each other
    vmax: float = 0.9

def accel_velocity_tracking(x: np.ndarray, v: np.ndarray, Vsat_at_x: np.ndarray, p: VTParams) -> np.ndarray:
    """
    vdot = ( kv*(Vsat - v) + sum frep - kd*v )/m
    """
    Fv = p.kv * (Vsat_at_x - v)
    Frep = repulsion_forces(x, p.krep, p.Rsafe)
    Fd = -p.kd * v
    a = (Fv + Frep + Fd) / p.m
    return a

def rk4_step_vt(x: np.ndarray, v: np.ndarray, dt: float, Vsat_at_x: np.ndarray, p: VTParams) -> Tuple[np.ndarray, np.ndarray]:
    def f(state_x, state_v, Vx):
        vx = saturate(state_v, p.vmax)
        ax = accel_velocity_tracking(state_x, state_v, Vx, p)
        return vx, ax

    k1x, k1v = f(x, v, Vsat_at_x)
    k2x, k2v = f(x + 0.5*dt*k1x, v + 0.5*dt*k1v, Vsat_at_x)
    k3x, k3v = f(x + 0.5*dt*k2x, v + 0.5*dt*k2v, Vsat_at_x)
    k4x, k4v = f(x + dt*k3x, v + dt*k3v, Vsat_at_x)

    x_next = x + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
    v_next = v + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return x_next, v_next


# =========================
# Optical flow -> velocity field
# =========================

def compute_optical_flow_frames(video_path: str, max_frames: int = 200, resize_w: int = 320) -> Tuple[np.ndarray, float]:
    """
    Returns flows as (T-1, H, W, 2) in pixel/frame units, plus fps.
    Uses Farneback optical flow for simplicity.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        scale = resize_w / float(w)
        new_h = int(round(h * scale))
        gray = cv2.resize(gray, (resize_w, new_h), interpolation=cv2.INTER_AREA)

        frames.append(gray)
    cap.release()

    if len(frames) < 2:
        raise ValueError("Video too short for optical flow.")

    flows = []
    for t in range(len(frames)-1):
        prev = frames[t]
        nxt = frames[t+1]
        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt, None,
            pyr_scale=0.5, levels=3, winsize=25,
            iterations=3, poly_n=7, poly_sigma=1.5, flags=0
        )
        flows.append(flow)
    flows = np.stack(flows, axis=0)  # (T-1,H,W,2)
    return flows, fps

def flow_to_world_velocity(flow: np.ndarray, world_box: Tuple[float, float, float, float], dt_video: float) -> np.ndarray:
    """
    Convert pixel/frame flow to world units/sec velocity field on the flow grid.
    This returns a velocity field in world coords per grid cell.
    """
    xmin, xmax, ymin, ymax = world_box
    H, W, _ = flow.shape

    # pixels to normalized
    # dx_norm per frame = flow_x / (W-1)
    # dy_norm per frame = flow_y / (H-1)
    fx = flow[..., 0] / max(1, (W - 1))
    fy = flow[..., 1] / max(1, (H - 1))

    # normalized to world
    vx = fx * (xmax - xmin) / dt_video
    vy = -fy * (ymax - ymin) / dt_video  # invert y
    V = np.stack([vx, vy], axis=-1)
    return V

def sample_velocity_field_at_positions(V: np.ndarray, x: np.ndarray, world_box: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Bilinear sample a grid velocity field V(H,W,2) at drone positions x(N,2).
    """
    xmin, xmax, ymin, ymax = world_box
    H, W, _ = V.shape

    # map world x,y -> normalized u,v in [0,1]
    u = (x[:, 0] - xmin) / (xmax - xmin + 1e-12)
    v = (ymax - x[:, 1]) / (ymax - ymin + 1e-12)

    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    px = u * (W - 1)
    py = v * (H - 1)

    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    sx = (px - x0).reshape(-1, 1)
    sy = (py - y0).reshape(-1, 1)

    V00 = V[y0, x0]
    V10 = V[y0, x1]
    V01 = V[y1, x0]
    V11 = V[y1, x1]

    V0 = V00 * (1 - sx) + V10 * sx
    V1 = V01 * (1 - sx) + V11 * sx
    Vxy = V0 * (1 - sy) + V1 * sy
    return Vxy


# =========================
# Visualization
# =========================

def animate_trajectory(traj: np.ndarray, world_box: Tuple[float, float, float, float],
                       out_mp4: str, title: str = "", fps: int = 30, dot_size: int = 12) -> None:
    xmin, xmax, ymin, ymax = world_box
    T, N, _ = traj.shape

    # Calculate actual bounds from trajectory for better framing
    x_tr = traj[:, :, 0].ravel()
    y_tr = traj[:, :, 1].ravel()
    xlo = min(x_tr.min(), xmin)
    xhi = max(x_tr.max(), xmax)
    ylo = min(y_tr.min(), ymin)
    yhi = max(y_tr.max(), ymax)
    # Add small padding
    pad = 0.1
    xlo -= pad
    xhi += pad
    ylo -= pad
    yhi += pad

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title, color="white", fontsize=10)
    ax.axis("off")

    # Simple glow effect for better visibility
    scat = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=dot_size, 
                      c="white", alpha=0.9, edgecolors="none")

    def update(frame):
        scat.set_offsets(traj[frame])
        return scat,

    ani = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)

    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800, codec='libx264',
                             extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
        ani.save(out_mp4, writer=writer)
        print(f"Created video: {out_mp4}")
    except Exception as e:
        print(f"FFMpegWriter failed ({e}), trying fallback...")
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(out_mp4, writer=writer)
        print(f"Created video: {out_mp4}")
    plt.close(fig)


# =========================
# Main pipeline
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word_image", default="inputs/word.png")
    ap.add_argument("--text_image", default="inputs/happy_new_year.png", help="Image file for 'Happy New Year' text (required)")
    ap.add_argument("--video", default="inputs/video.mp4")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--N", type=int, default=400, help="Number of particles (higher = clearer formation)")
    ap.add_argument("--seed", type=int, default=7)

    # world / simulation
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--steps1", type=int, default=1200, help="Simulation steps for stage1 (more = better convergence)")
    ap.add_argument("--steps2", type=int, default=1500, help="Simulation steps for stage2 (more = better convergence)")
    ap.add_argument("--max_video_frames", type=int, default=220)

    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    # Pick a world box where all shapes fit nicely
    world_box = (-1.2, 1.2, -0.6, 0.6)

    # ---------- Initial state ----------
    x0 = init_positions_grid(args.N, box=world_box, jitter=0.01)
    v0 = np.zeros_like(x0)

    # ---------- Stage 1: handwritten word ----------
    pts01 = load_handwritten_word_points(args.word_image, n_points=args.N)
    T1 = pts01_to_world(pts01, world_box=world_box)
    T1 = assign_targets_nearest(x0, T1)

    # Tuned parameters for better handwritten word formation
    # Repulsion ensures collision-free behavior: krep and Rsafe prevent drones from getting too close
    p1 = SwarmParams(kp=12.0, kd=5.0, krep=0.008, Rsafe=0.03, vmax=1.0)
    traj1, x1, v1 = simulate_to_targets(x0, v0, T1, p1, dt=args.dt, steps=args.steps1)
    np.save(os.path.join(args.outdir, "traj_stage1.npy"), traj1)

    # ---------- Stage 2: greeting text ----------
    # Use provided image file for text (required)
    if not os.path.exists(args.text_image):
        raise FileNotFoundError(f"Text image not found: {args.text_image}. Please provide an image file for 'Happy New Year' text.")
    print(f"Using image file for text: {args.text_image}")
    pts01_2 = load_handwritten_word_points(args.text_image, n_points=args.N)
    
    # Use full world box for text (no inset needed with proper normalization)
    T2 = pts01_to_world(pts01_2, world_box=world_box)
    
    # Use same assignment and similar parameters as Stage 1 (which works well)
    T2 = assign_targets_nearest(x1, T2, preserve_order=False)

    # Use similar parameters to Stage 1 - they work well for word formation
    # Slightly higher repulsion to handle the more complex multi-word text
    # Collision-free: repulsion prevents drones from colliding
    p2 = SwarmParams(kp=12.0, kd=5.0, krep=0.010, Rsafe=0.035, vmax=1.0)
    traj2, x2, v2 = simulate_to_targets(x1, v1, T2, p2, dt=args.dt, steps=args.steps2)
    np.save(os.path.join(args.outdir, "traj_stage2.npy"), traj2)

    # ---------- Stage 3: velocity tracking from video ----------
    if not os.path.exists(args.video):
        print(f"Warning: Video file not found: {args.video}. Skipping Stage 3.")
        traj3 = traj2  # Use Stage 2 trajectory as fallback
    else:
        try:
            flows, fps_video = compute_optical_flow_frames(args.video, max_frames=args.max_video_frames, resize_w=320)
            dt_video = 1.0 / float(fps_video if fps_video and fps_video > 1 else 25.0)

            p3 = VTParams()

            x = x2.copy()
            v = v2.copy()

            traj3 = [x.copy()]
            for t in range(flows.shape[0]):
                Vworld = flow_to_world_velocity(flows[t], world_box=world_box, dt_video=dt_video)

                # Sample velocity at drone positions
                Vx = sample_velocity_field_at_positions(Vworld, x, world_box=world_box)

                # Saturate the reference velocity field (per spec)
                Vx = saturate(Vx, p3.vmax)

                # advance a few sim substeps per video frame (smoother)
                substeps = 2
                dt = dt_video / substeps
                for _ in range(substeps):
                    x, v = rk4_step_vt(x, v, dt, Vx, p3)

                traj3.append(x.copy())
        except Exception as e:
            print(f"Error in Stage 3: {e}. Using Stage 2 trajectory as fallback.")
            traj3 = traj2

    traj3 = np.stack(traj3, axis=0)
    np.save(os.path.join(args.outdir, "traj_stage3.npy"), traj3)

    # ---------- Visualizations ----------
    # Note: requires ffmpeg installed for mp4 saving
    animate_trajectory(traj1, world_box, os.path.join(args.outdir, "stage1.mp4"),
                       title="Stage 1: Handwritten formation", fps=30, dot_size=14)
    animate_trajectory(traj2, world_box, os.path.join(args.outdir, "stage2.mp4"),
                       title="Stage 2: Transition to HAPPY NEW YEAR!", fps=30, dot_size=16)
    animate_trajectory(traj3, world_box, os.path.join(args.outdir, "stage3.mp4"),
                       title="Stage 3: Velocity tracking + repulsion", fps=30, dot_size=14)

    print("DONE.")
    print("Saved trajectories + videos into:", args.outdir)


if __name__ == "__main__":
    main()
