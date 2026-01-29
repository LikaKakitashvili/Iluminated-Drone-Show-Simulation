# Illuminated Drone Show Simulation — Presentation Guide

This document explains the program structure for a presentation split among **three people**, each responsible for **one stage** and **three subtasks** within that stage.

---

## Overview

The simulation has **three main stages**:

| Stage | Description | Output |
|-------|-------------|--------|
| **Stage 1** | Drones form a **handwritten word** from an image | `stage1.mp4` |
| **Stage 2** | Drones **transition** to form **"Happy New Year"** text | `stage2.mp4` |
| **Stage 3** | Drones **track a moving object** from a video | `stage3.mp4` |

Each stage uses: **(1)** turning an input (image or video) into target points, **(2)** assigning drones to those targets and optionally simulating, **(3)** running physics and/or rendering the animation.

---

## Person 1 — Stage 1: Handwritten Formation

**Goal:** Drones start in a grid and move to form the handwritten word from `inputs/handwritten.jpeg`.

### Subtask 1.1 — Image to target points

**What it does:** Load the handwritten image, binarize it, sample a set of 2D points that follow the word’s shape (contours + fill), remove tiny isolated clusters, and normalize to `[0, 1]`.

**Functions:**

| Function | Role |
|----------|------|
| `load_handwritten_word_points(image_path, n_points)` | Main entry: read image, threshold, morphology, connected components, then contour + fill sampling and normalization. |
| `normalize_pts01_tight(pts01, margin)` | Normalize points to [0,1] using a tight bounding box plus margin. |
| `sample_points_from_contours(mask, n_points, min_contour_area)` | Sample points along contour arc-length; ignores small contours. |
| `sample_points_from_binary_mask(mask, n_points, mode)` | Sample from edges (`"edge"`) or filled region (`"fill"`) using farthest-point sampling. |
| `farthest_point_sampling(points, n_samples)` | Greedy farthest-point sampling for more uniform distribution. |
| `remove_singleton_cluster_points(pts01, min_cluster_size, connectivity_radius)` | Remove points in very small clusters (e.g. 4-point speckles) via union-find. |

**In `main()`:**  
`pts01 = load_handwritten_word_points(...)` → `pts01 = remove_singleton_cluster_points(...)`.

---

### Subtask 1.2 — World mapping and drone–target assignment

**What it does:** Map normalized points into world coordinates, create initial drone positions on a grid, and assign each drone to one target (nearest unused target).

**Functions:**

| Function | Role |
|----------|------|
| `pts01_to_world(pts01, world_box)` | Map [0,1]×[0,1] points to world box `(xmin, xmax, ymin, ymax)`. |
| `init_positions_grid(N, box, jitter)` | Place N drones on a 2D grid inside the box with small random jitter. |
| `assign_targets_nearest(x, T, preserve_order)` | Assign each drone to a distinct target by greedy nearest-neighbor; optional proportional ordering. |

**In `main()`:**  
`N_effective = pts01.shape[0]`, `x0 = init_positions_grid(N_effective, ...)`, `T1 = pts01_to_world(pts01, ...)`, `T1 = assign_targets_nearest(x0, T1)`.

---

### Subtask 1.3 — Swarm simulation and animation

**What it does:** Simulate drones with PD control plus repulsion (collision avoidance), integrate with RK4, stop when close to targets, then render the trajectory as a video.

**Functions:**

| Function | Role |
|----------|------|
| `SwarmParams` (dataclass) | Parameters: `m`, `kp`, `kd`, `krep`, `Rsafe`, `vmax`. |
| `saturate(v, vmax)` | Clamp velocity magnitude to `vmax`. |
| `repulsion_forces(x, krep, Rsafe)` | Pairwise repulsion when distance &lt; `Rsafe`; returns (N,2) force. |
| `accel_position_tracking(x, v, T, p)` | Acceleration: `kp*(T−x) + repulsion − kd*v` over mass. |
| `rk4_step(x, v, dt, T, p)` | One RK4 step for positions and velocities. |
| `simulate_to_targets(x0, v0, T, p, dt, steps, ...)` | Run simulation until convergence or max steps; returns trajectory and final state. |
| `animate_trajectory(traj, world_box, out_mp4, title, fps, ...)` | Build matplotlib animation and save as MP4. |

**In `main()`:**  
`p1 = SwarmParams(...)`, `traj1, x1, v1 = simulate_to_targets(x0, v0, T1, p1, ...)`, `animate_trajectory(traj1, ..., "stage1.mp4", "Stage 1: Handwritten formation", ...)`.

---

## Person 2 — Stage 2: Transition to Happy New Year

**Goal:** From the end state of Stage 1, drones move to form the “Happy New Year” text from `inputs/happy_new_year.png`.

### Subtask 2.1 — Text image to target points

**What it does:** Load the greeting image, treat it like a “handwritten” image (same pipeline: binarize, contours + fill sampling, normalize) to get a point set of size `N_effective`.

**Functions:**

| Function | Role |
|----------|------|
| `load_handwritten_word_points(image_path, n_points)` | Same as Stage 1: binarize, contours, fill, normalize. |
| `normalize_pts01_tight(pts01, margin)` | Tight normalization to [0,1]. |
| `sample_points_from_contours(mask, n_points, min_contour_area)` | Contour-based sampling. |
| `sample_points_from_binary_mask(mask, n_points, mode)` | Edge or fill sampling with farthest-point. |
| `farthest_point_sampling(points, n_samples)` | Uniform-ish sampling. |

**In `main()`:**  
`pts01_2 = load_handwritten_word_points(args.text_image, n_points=N_effective)`, then `T2 = pts01_to_world(pts01_2, ...)`.

---

### Subtask 2.2 — Transition assignment

**What it does:** Use final positions and velocities from Stage 1 (`x1`, `v1`). Map text points to world and assign each drone to a target in the new formation (nearest-target assignment).

**Functions:**

| Function | Role |
|----------|------|
| `pts01_to_world(pts01, world_box)` | Image points → world coordinates. |
| `assign_targets_nearest(x, T, preserve_order)` | Assign drones (here `x1`) to targets `T2`; `preserve_order=False` for this stage. |

**In `main()`:**  
`T2 = pts01_to_world(pts01_2, world_box)`, `T2 = assign_targets_nearest(x1, T2, preserve_order=False)`.

---

### Subtask 2.3 — Simulation and animation

**What it does:** Run the same swarm dynamics (PD + repulsion, RK4) from `x1`, `v1` toward `T2`, then animate the trajectory.

**Functions:**

| Function | Role |
|----------|------|
| `SwarmParams` | Tuned parameters for stage 2 (e.g. slightly higher repulsion). |
| `simulate_to_targets(x1, v1, T2, p2, ...)` | Simulate from Stage 1 end state to greeting targets. |
| `rk4_step`, `accel_position_tracking`, `repulsion_forces`, `saturate` | Used inside simulation. |
| `animate_trajectory(traj2, ..., "stage2.mp4", "Stage 2: Transition to HAPPY NEW YEAR!", ...)` | Render Stage 2 video. |

**In `main()`:**  
`p2 = SwarmParams(...)`, `traj2, x2, v2 = simulate_to_targets(x1, v1, T2, p2, ...)`, `animate_trajectory(traj2, ...)`.

---

## Person 3 — Stage 3: Video object tracking

**Goal:** Drones transition from the Stage 2 formation to the first visible object in a video, then track the object’s changing shape frame by frame.

### Subtask 3.1 — Video to object masks and time-varying shapes

**What it does:** Read video, build a background model, subtract it to get a moving-object mask per frame, then for each mask sample `n_points` and convert to world-coordinate shapes (list of (N,2) arrays per frame).

**Functions:**

| Function | Role |
|----------|------|
| `extract_object_from_video(video_path, max_frames, resize_w)` | Median background over first frames; per-frame diff + threshold + morphology; keep largest contour per frame; return list of binary masks and fps. |
| `extract_object_shapes_from_masks(masks, n_points, world_box)` | For each mask: sample points (contours or binary mask), map to world; return list of (N,2) shapes. |
| `sample_points_from_contours(mask, n_points)` | Points along contour; fallback used inside shape extraction. |
| `sample_points_from_binary_mask(mask, n_points, mode)` | Fallback when contour sampling fails. |
| `pts01_to_world(pts01, world_box)` | Normalized points → world box. |

**In `main()`:**  
`masks, fps_video = extract_object_from_video(...)`, `object_shapes = extract_object_shapes_from_masks(masks, n_points=N_effective, world_box=world_box)`.

---

### Subtask 3.2 — Transition to object and shape-preserving assignment

**What it does:** From Stage 2 end state (`x2`, `v2`), assign drones to the first valid object shape, run a short simulation (transition). Then, for each subsequent video frame, assign targets so that each drone keeps a similar *relative* position within the shape (direction and distance from centroid), avoiding big jumps.

**Functions:**

| Function | Role |
|----------|------|
| `assign_targets_nearest(x, T, preserve_order)` | Used for the one-off transition from greeting to first object shape. |
| `assign_targets_shape_preserving(x, T_prev, T_curr)` | For each drone, find best target in `T_curr` that matches its relative position in `T_prev` (centered + scaled); minimizes shape distortion. |
| `simulate_to_targets(x2, v2, T_transition, p_transition, ...)` | Short run to move from greeting to first object shape. |
| `rk4_step(x, v, dt, T, p)` | Single integration step used in the tracking loop. |
| `SwarmParams` | Used for transition and tracking (e.g. `p_transition`, `p3`). |

**In `main()`:**  
`T_transition = assign_targets_nearest(x2, T_object_initial, ...)`, `traj_transition, x_transition, v_transition = simulate_to_targets(...)`; then in the loop: `T_assigned = assign_targets_shape_preserving(x, T_prev, T_curr)` and multiple `rk4_step(...)` per frame.

---

### Subtask 3.3 — Frame-by-frame tracking simulation and animation

**What it does:** For each video frame (after the first valid shape), update assigned targets with shape-preserving assignment, advance the swarm with RK4 for one frame’s worth of time, append to trajectory. Concatenate transition + tracking trajectory and render Stage 3 video.

**Functions:**

| Function | Role |
|----------|------|
| `rk4_step(x, v, dt_step, T_assigned, p3)` | Advance positions and velocities one sub-step (multiple per video frame to match fps). |
| `assign_targets_shape_preserving(x, T_prev, T_curr)` | Compute target assignment for current frame. |
| `animate_trajectory(traj3, ..., "stage3.mp4", "Stage 3: Velocity tracking + repulsion", ...)` | Render full Stage 3 trajectory (transition + tracking). |

**In `main()`:**  
Loop over `object_shapes`: `T_assigned = assign_targets_shape_preserving(...)`, then `for _ in range(substeps): x, v = rk4_step(...)`, `traj3_tracking.append(x.copy())`. Then `traj3 = np.vstack([traj_transition, traj3_tracking])`, `animate_trajectory(traj3, ...)`.

---

## Shared / utility functions

Used in more than one stage or as helpers:

| Function | Used in | Purpose |
|----------|---------|--------|
| `ensure_dir(path)` | main | Create output directory. |
| `set_seed(seed)` | main | Reproducible randomness. |
| `normalize_pts01_tight` | Stage 1, Stage 2 (inside load) | Normalize point sets. |
| `normalize_pts01_tight_aspect` | (render_text_to_points, if used) | Aspect-preserving normalization. |
| `pts01_to_world` | All stages | Image [0,1] → world box. |
| `assign_targets_nearest` | All stages | Drone–target assignment. |
| `repulsion_forces` | All stages (inside accel) | Collision avoidance. |
| `animate_trajectory` | All stages | Trajectory → MP4. |

---

## Quick reference: who presents what

| Person | Stage | Subtask 1 | Subtask 2 | Subtask 3 |
|--------|--------|-----------|------------|-----------|
| **1** | Handwritten formation | Image → points (load, sample, remove singletons) | World mapping + assignment (grid, T1, assign) | Swarm simulation + animation (physics, RK4, stage1.mp4) |
| **2** | Happy New Year | Text image → points (load, sample) | Transition assignment (x1→T2) | Simulation + animation (stage2.mp4) |
| **3** | Video tracking | Video → masks → shapes | Transition + shape-preserving assignment | Frame-by-frame tracking + animation (stage3.mp4) |

---

## File and data flow

```
inputs/handwritten.jpeg     → Stage 1 → outputs/stage1.mp4, traj_stage1.npy
inputs/happy_new_year.png   → Stage 2 → outputs/stage2.mp4, traj_stage2.npy
inputs/video.mp4            → Stage 3 → outputs/stage3.mp4, traj_stage3.npy
```

State is passed sequentially: **x0, v0** (grid) → Stage 1 → **x1, v1** → Stage 2 → **x2, v2** → Stage 3 (transition + tracking) → **traj3**.
