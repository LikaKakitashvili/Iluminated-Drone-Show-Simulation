# Illuminated Drone Show Simulation — Presentation Notes

This document separates the **three subtasks** of the project, summarizes implementation and **mathematical models**, explains **why** they were used, and gives short answers to likely **questions** so you can defend the design in the presentation.

**How to use:** Before the presentation, read each subtask’s “What this subtask does” and “Mathematical models.” During Q&A, use “Questions and answers” for that subtask plus “Common questions” at the end. The “Quick reference” table is for a final recap slide.

---

## Project overview

**Goal:** Simulate a swarm of drones (particles) that form shapes in three stages:

1. **Stage 1** — From a grid to a **handwritten word** (from an image).
2. **Stage 2** — From that word to a **greeting text** (e.g. “Happy New Year”) from a second image.
3. **Stage 3** — From the greeting to **tracking a moving object** extracted from a video over time.

**Output:** Three MP4 videos (`stage1.mp4`, `stage2.mp4`, `stage3.mp4`) and trajectory arrays (`.npy`).

**Pipeline (high level):** For each stage we (1) get **target points** from the input (image or video), (2) **assign** each drone to a target, (3) **simulate** dynamics (PD control + repulsion) with RK4, (4) **animate** the trajectory.

---

# Subtask 1: Handwritten word formation (Stage 1)

## What this subtask does

- Takes a **handwritten word image** (e.g. `inputs/handwritten.jpeg`).
- Produces a set of **target points** that follow the word’s shape.
- Simulates drones from an **initial grid** to those targets.
- Output: formation that spells the word and video `stage1.mp4`.

## Implementation (main functions)

| Function / step | Role |
|-----------------|------|
| `load_handwritten_word_points(image_path, n_points)` | Load image, binarize, morphology, connected components, then **contour + fill** sampling and aspect-preserving normalization. |
| `_preprocess_mask_like_load(img)` | Blur, adaptive/Otsu threshold, open+dilate, keep components ≥ 3% of max area. |
| `sample_points_from_contours(mask, n_points, aspect_preserve=True)` | Sample points along contour **arc-length**; normalize with single scale (max of width, height) so aspect ratio is preserved. |
| `sample_points_from_binary_mask(mask, n_points, mode="fill")` | Sample from **filled** foreground with **farthest-point sampling**. |
| `farthest_point_sampling(points, n_samples)` | Greedy farthest-point sampling for **uniform** coverage. |
| `normalize_pts01_tight_aspect(pts01, margin)` | Normalize to [0,1]-like range with **one scale** so aspect ratio of the word is preserved. |
| `remove_singleton_cluster_points(pts01, ...)` | Remove tiny clusters (e.g. 4-point speckles) via **union-find** so only the main word remains. |
| `pts01_to_world_aspect_preserving(pts01, world_box, box_inset)` | Map normalized points to world coordinates with **one scale** and optional inset so the word does not touch the edges. |
| `init_positions_grid(N, box, jitter)` | Initial drone positions on a **grid** with small random jitter. |
| `assign_targets_nearest(x0, T1)` | Assign each drone to a **unique** target by **greedy nearest** assignment. |
| `simulate_to_targets(x0, v0, T1, p1, dt, steps)` | Run **RK4** with PD control and repulsion until convergence or max steps. |

## Mathematical models (Subtask 1)

### 1. Point sampling (contours)

- **Contours:** From binary mask with `cv2.findContours` (external). Filter: ≥ 5 points and area ≥ `min_contour_area`.
- **Arc-length:** For each contour, perimeter \(L = \sum \|p_{i+1} - p_i\|\). Allocate \(n_k = \lfloor n_{\text{points}} \cdot (L_k / L_{\text{total}}) \rfloor\) points to contour \(k\).
- **Uniform along contour:** For contour \(k\), cumulative length \(s_j\); sample at \(s = 0, L_k/n_k, 2L_k/n_k, \ldots\); interpolate linearly between vertices. So points are **uniform in arc-length**, which keeps letter strokes well defined.

### 2. Farthest-point sampling

- Start from a random point; then repeatedly add the point that is **farthest** from the current set (in Euclidean distance).
- **Why:** Spreads points evenly over the shape and avoids clustering; better than random sampling for readability of the formation.

### 3. Aspect-preserving normalization

- Pixel coords \((x,y)\) with image size \((w,h)\). Use **one scale** \(S = \max(w-1, h-1)\): \(x' = x/S\), \(y' = y/S\).
- **Why:** If we used \(x' = x/(w-1)\) and \(y' = y/(h-1)\), a wide image would be squeezed into a square and the word would look stretched vertically. One scale keeps the **true aspect ratio** of the handwriting.

### 4. PD control + repulsion dynamics

- **Position error:** \(\mathbf{e} = \mathbf{T}_i - \mathbf{x}_i\) (target minus position).
- **Proportional force:** \(\mathbf{F}_p = k_p \mathbf{e}\).
- **Repulsion (pairwise):** If \(\|\mathbf{x}_i - \mathbf{x}_j\| < R_{\text{safe}}\),  
  \(\mathbf{F}_{\text{rep}}^{ij} = k_{\text{rep}} \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|^3}\) (inverse-cube so it blows up near contact).
- **Damping:** \(\mathbf{F}_d = -k_d \mathbf{v}_i\).
- **Acceleration:** \(\ddot{\mathbf{x}}_i = (\mathbf{F}_p + \sum_j \mathbf{F}_{\text{rep}}^{ij} + \mathbf{F}_d) / m\).
- **Velocity saturation:** \(\mathbf{v} \mapsto \min(1, v_{\max}/\|\mathbf{v}\|)\,\mathbf{v}\).
- **Why PD:** Simple, stable, and drives each drone to its target; repulsion avoids collisions; damping avoids overshoot and oscillation.

### 5. RK4 integration

- \(\dot{\mathbf{x}} = \mathbf{v}\), \(\dot{\mathbf{v}} = \mathbf{a}(\mathbf{x},\mathbf{v},\mathbf{T})\).
- Standard **Runge–Kutta 4** with step `dt` to advance \((\mathbf{x}, \mathbf{v})\).
- **Why RK4:** Fourth-order accuracy and stable for this smooth dynamics; better than Euler for the same `dt`.

### 6. Nearest-target assignment

- For each drone \(i\), choose target \(j\) that minimizes \(\|\mathbf{x}_i - \mathbf{T}_j\|\) among **unused** targets; mark \(j\) as used. Repeat.
- **Why:** Minimizes total distance to targets and gives a one-to-one assignment; simple and works well for static formations.

---

## Why these choices (Subtask 1)

- **Contour + fill:** Contours give sharp edges (readable letters); fill adds body so the word is not only an outline.
- **Aspect-preserving norm + world mapping:** So the word on screen has the **same proportions** as in the original image.
- **Singleton removal:** Small connected components (e.g. dots, noise) become tiny clusters of points; removing them avoids stray “blobs” in the formation.
- **Repulsion:** Prevents drones from stacking on the same target and avoids unrealistic overlap.
- **Early stopping:** When all drones are within a small distance of their targets, we stop; saves time and keeps the video length reasonable.

---

## Questions and answers (Subtask 1)

**Q: Why not random sampling instead of farthest-point / contour?**  
A: Random sampling clusters in dense regions and leaves gaps; the word would look noisy. Contour + farthest-point gives **uniform** coverage and preserves **structure** (strokes, edges).

**Q: What is the role of \(R_{\text{safe}}\) and \(k_{\text{rep}}\)?**  
A: \(R_{\text{safe}}\) is the distance below which two drones repel. \(k_{\text{rep}}\) sets the strength; too high causes jitter, too low allows overlap. We tune them so the formation is collision-free and smooth.

**Q: Why RK4 instead of Euler?**  
A: RK4 is fourth-order accurate; for the same `dt` we get much smaller error and can use larger steps. Euler would need smaller `dt` for similar accuracy, so more steps and longer runtime.

**Q: Why normalize with a single scale (max of w, h)?**  
A: So that one pixel in \(x\) and one pixel in \(y\) correspond to the same “physical” length in normalized space. That preserves the **aspect ratio** of the handwritten word.

**Q: What does “box_inset” do?**  
A: We map the formation into a **smaller** inner part of the world box (e.g. 10% margin). So the word does not sit exactly on the boundary and we avoid numerical/visual edge effects.

**Q: How does “singleton cluster removal” work?**  
A: We use **union-find**: two points are in the same cluster if they are within a small distance. We then remove clusters smaller than a minimum size (e.g. 5 points). That removes noise blobs (e.g. a dot of 4 points) and keeps only the main word.

**Q: Why velocity saturation (cap on speed)?**  
A: Without a cap, a drone far from its target can reach very high speed and overshoot or cause numerical issues. We cap velocity at \(v_{\max}\); the formation still converges but motion stays stable.

---

# Subtask 2: Transition to greeting text (Stage 2)

## What this subtask does

- Takes the **final state** of Stage 1 (positions \(\mathbf{x}_1\), velocities \(\mathbf{v}_1\)).
- Loads a **second image** (e.g. “Happy New Year”), gets target points \(\mathbf{T}_2\) with the **same pipeline** as Stage 1 (contour + fill, aspect-preserving).
- Assigns drones to \(\mathbf{T}_2\) (nearest-neighbor).
- Simulates from \((\mathbf{x}_1, \mathbf{v}_1)\) to \(\mathbf{T}_2\).
- Output: formation showing the greeting text and video `stage2.mp4`.

## Implementation (main functions)

| Function / step | Role |
|----------------|------|
| `load_handwritten_word_points(text_image, n_points=N_effective)` | Same as Stage 1: binarize, contour+fill, aspect-preserving normalization. |
| `pts01_to_world_aspect_preserving(pts01_2, world_box)` | Map greeting points to world with same aspect and inset as Stage 1. |
| `assign_targets_nearest(x1, T2, preserve_order=False)` | Assign each drone to a target in the new formation (greedy nearest). |
| `simulate_to_targets(x1, v1, T2, p2, dt, steps)` | Same dynamics as Stage 1; different \(k_p, k_d, k_{\text{rep}}, R_{\text{safe}}\) if tuned. |

## Mathematical models (Subtask 2)

- **Same as Stage 1:** PD control, repulsion, RK4, velocity saturation.
- **Assignment:** Again greedy nearest-neighbor; we do **not** use “preserve order” so that drones can move freely to the new text layout.
- **Continuity:** We start from \((\mathbf{x}_1, \mathbf{v}_1)\) so the motion is **continuous** from the end of Stage 1; no re-initialization.

## Why these choices (Subtask 2)

- **Same image pipeline as Stage 1:** Consistent quality and aspect ratio for any text image.
- **Nearest assignment (no order preserve):** The greeting layout can be very different from the handwritten word; forcing order would create long, unnatural paths. Nearest-neighbor minimizes total distance and looks natural.
- **Same dynamics:** One unified physics model for all static-target stages; only parameters may change.

## Questions and answers (Subtask 2)

**Q: Why start from Stage 1’s end state instead of a new grid?**  
A: To show a **continuous show**: drones move from the first word **into** the second. Starting from a grid again would look like a new scene, not a transition.

**Q: Why not preserve order when assigning to T2?**  
A: The two shapes can have different topology and layout. Preserving order would force drones from the left side of the first word to the left side of the second, which can be far and create crossing paths. Nearest-neighbor gives shorter, clearer motion.

**Q: Could we use a different number of points for the greeting?**  
A: We use \(N_{\text{effective}}\) (same as Stage 1 after singleton removal) so we have **one target per drone** and no leftover drones.

---

# Subtask 3: Video object tracking (Stage 3)

## What this subtask does

- **Extracts** a moving object from a video (background subtraction → binary mask per frame).
- Builds **one set of target points per frame** from the mask (contour sampling, fixed world box for Stage 3).
- **Transitions** drones from the end of Stage 2 to the first object shape.
- **Tracks** the object over time: each frame, assign targets with **shape-preserving** assignment, then advance dynamics.
- Output: formation following the moving object and video `stage3.mp4`.

## Implementation (main functions)

| Function / step | Role |
|----------------|------|
| `extract_object_from_video(video_path, max_frames, resize_w)` | Median background from first frames; per-frame diff + threshold + morphology; keep **largest** contour per frame → list of masks + fps. |
| `extract_object_shapes_from_masks(masks, n_points, world_box)` | For each mask: **Stage-3-only** contour sampling (`_sample_points_from_mask_stage3`) with [0,1]×[0,1] normalization; map to world with `pts01_to_world`. |
| `_sample_points_from_mask_stage3(mask, n_points)` | Contour sampling with normalization \(x' = x/(w-1)\), \(y' = y/(h-1)\) so shapes live in a fixed world box (same behaviour as original design). |
| `transform_points_between_boxes`, `transform_velocities_between_boxes` | Map Stage 2 end state from main `world_box` into **Stage 3 fixed box** \((-1.2, 1.2, -0.6, 0.6)\); after simulation, map trajectory back for consistent video scale. |
| `assign_targets_shape_preserving(x, T_prev, T_curr)` | Assign drones to **current** targets so each drone keeps a similar **relative position** in the shape (direction + relative distance from centroid). |
| `rk4_step(x, v, dt, T_assigned, p3)` | Same PD + repulsion + RK4; run for one frame’s worth of time (substeps). |

## Mathematical models (Subtask 3)

### 1. Background subtraction

- **Background:** Median of the first ~30 frames (in gray). Assumes the object is not dominant in that window.
- **Foreground:** \(|I_t - B| > \theta\) (e.g. \(\theta = 30\)), then morphology (open/close) to remove noise and fill holes.
- **Object:** Largest connected component per frame; its contour is the object outline.
- **Why median:** Robust to short occlusions and small movements; simple and fast.

### 2. Shape from mask (Stage 3)

- **Contours** from the binary mask (external); filter by area and point count.
- **Arc-length sampling** along contours (same idea as Stage 1) to get \(n\) points per frame.
- **Normalization:** \(x' = x/(w-1)\), \(y' = y/(h-1)\) so points are in \([0,1]\times[0,1]\) (Stage 3 uses a fixed world box, so we keep this convention for consistent behaviour).
- **World:** Linear map from \([0,1]^2\) to the Stage 3 box \((-1.2, 1.2, -0.6, 0.6)\).

### 3. Shape-preserving assignment

- **Idea:** Each drone should keep its “role” in the shape: same **direction** from centroid and similar **relative distance**.
- **Centroids:** \(\mathbf{c}_{\text{prev}} = \frac{1}{N}\sum \mathbf{T}_{\text{prev}}\), \(\mathbf{c}_{\text{curr}} = \frac{1}{N}\sum \mathbf{T}_{\text{curr}}\).
- **Normalize:** \(\tilde{\mathbf{T}}_{\text{prev}} = (\mathbf{T}_{\text{prev}} - \mathbf{c}_{\text{prev}}) / \sigma_{\text{prev}}\), \(\tilde{\mathbf{T}}_{\text{curr}} = (\mathbf{T}_{\text{curr}} - \mathbf{c}_{\text{curr}}) / \sigma_{\text{curr}}\) (scale by standard deviation).
- **For each drone \(i\):** Find its closest target in the **previous** frame → index \(p\). Take normalized position \(\tilde{\mathbf{r}}_p\).
- **Match in current frame:** Find unused target \(j\) that maximizes a **score** combining:
  - **Direction similarity:** \(\hat{\tilde{\mathbf{r}}}_p \cdot \hat{\tilde{\mathbf{r}}}_j\) (cosine similarity).
  - **Distance similarity:** \(1/(1 + |\|\tilde{\mathbf{r}}_p\| - \|\tilde{\mathbf{r}}_j\||)\).
  - Example: \(\text{score} = 0.7 \cdot \text{dir} + 0.3 \cdot \text{dist}\).
- **Why:** If we used only nearest-neighbor from current positions, drones would “jump” to the nearest target each frame and the formation would jitter. Shape-preserving keeps **smooth** motion and **coherent** roles (e.g. “nose” of the shape stays the same drones).

### 4. Dynamics (same as Stage 1/2)

- PD + repulsion + RK4 in the **Stage 3 box**; then trajectory is transformed back to the main `world_box` for saving and animation.

### 5. Fixed world box for Stage 3

- Stage 3 runs in **fixed** box \((-1.2, 1.2, -0.6, 0.6)\) so scaling and physics match the original design.
- Stage 2 end state \((\mathbf{x}_2, \mathbf{v}_2)\) is **mapped** into this box; after simulation, the trajectory is **mapped back** to the main (possibly scaled) `world_box` so all three videos share the same visual scale.

---

## Why these choices (Subtask 3)

- **Median background:** Simple, robust, no need for learning; works well when the object moves and the background is mostly static.
- **Largest contour only:** Assumes **one** main object; avoids mixing several objects into one shape.
- **Shape-preserving assignment:** Reduces jitter and keeps the formation stable while the object deforms or moves.
- **Contour-only for Stage 3:** Original design sampled only the outline; we keep that for Stage 3 so behaviour matches the reference implementation; Stages 1 and 2 still use contour+fill for better readability.
- **Fixed box + transform:** Keeps Stage 3 physics and scale consistent with the original; the transform makes the final video consistent with Stages 1 and 2 when using `spacing_scale`.

---

## Questions and answers (Subtask 3)

**Q: Why shape-preserving instead of nearest-neighbor every frame?**  
A: Nearest-neighbor only minimizes current distance; it does not care about “who was where last frame.” So drones can swap roles every frame and the formation **jitters**. Shape-preserving tries to keep the same relative role (direction and distance from center), so motion is **smooth** and the formation looks coherent.

**Q: How is the score (0.7 direction + 0.3 distance) chosen?**  
A: Direction (angle) is usually more stable for “same part of the shape,” so we weight it higher (0.7). Distance from center helps distinguish inner vs outer points; 0.3 is enough without overriding direction. The weights can be tuned; 0.7/0.3 is a reasonable default.

**Q: Why use a fixed world box for Stage 3 and then transform?**  
A: So that Stage 3 **simulation** runs in the same coordinate system and scale as the original design (no dependence on `spacing_scale`). The transform back to the main `world_box` is only for **output** (saving and animation) so all three stages appear at the same scale.

**Q: Why median for the background?**  
A: Median over time is robust to the object moving through the scene (each pixel is “object” only in a few frames). Mean would be blurred by the object. More advanced methods (e.g. Gaussian mixture) could be used but median is simple and effective for this setting.

**Q: What if the object leaves the frame?**  
A: We keep the previous frame’s shape when the current mask is empty or too small; so the formation holds the last known shape. We could also fade out or switch to a default formation.

**Q: Why contour-only for Stage 3 and not contour+fill?**  
A: To match the **original** Stage 3 behaviour (outline tracking). Contour+fill could be added for a denser formation inside the object, but the current design keeps Stage 3 as outline-only for consistency with the reference.

---

# Common questions (all subtasks)

**Q: What is the order of the pipeline?**  
A: (1) Load input → (2) Binarize + morphology + connected components → (3) Sample points (contour ± fill) → (4) Normalize (aspect-preserving for 1/2, [0,1]² for 3) → (5) Map to world → (6) Assign drones to targets → (7) Simulate (RK4 + PD + repulsion) → (8) Animate and save.

**Q: Why PD and not P or PID?**  
A: P alone would overshoot and oscillate. D adds **damping** (dissipation), so the system settles without large overshoots. We don’t need integral (I) because the target is fixed per phase and there’s no steady-state offset.

**Q: Why inverse-cube repulsion \(\sim 1/d^3\)?**  
A: So that repulsion grows **strongly** as distance shrinks (strong near contact, small far away). Inverse-square would be weaker at small \(d\); inverse-cube keeps drones from overlapping while not affecting distant pairs much.

**Q: How do you choose \(k_p\), \(k_d\), \(k_{\text{rep}}\), \(R_{\text{safe}}\)?**  
A: By tuning: \(k_p\) large → fast approach but risk overshoot; \(k_d\) large → more damping, smoother; \(k_{\text{rep}}\) and \(R_{\text{safe}}\) so drones don’t overlap but don’t push each other too far. Values in the code are chosen so formations converge and look smooth.

**Q: What is “convergence” in simulate_to_targets?**  
A: We check periodically (e.g. every 50 steps): if **max** distance and **mean** distance of drones to their targets are below a threshold (e.g. 0.025), we **stop early**. So we don’t always run the full step count.

**Q: How many drones / points?**  
A: Controlled by `--N` (e.g. 400). After singleton removal in Stage 1 we get \(N_{\text{effective}}\); that number is used for Stages 2 and 3 so there is always one target per drone.

---

# Quick reference: three subtasks

| Subtask | Input | Main output | Key models |
|--------|--------|-------------|------------|
| **1. Handwritten formation** | Handwritten image | Word formation + `stage1.mp4` | Contour+fill sampling, aspect norm, PD+repulsion, RK4, nearest assignment |
| **2. Greeting transition** | End of Stage 1 + text image | Greeting formation + `stage2.mp4` | Same sampling/norm as Stage 1, nearest assignment, same dynamics |
| **3. Video tracking** | End of Stage 2 + video | Object-tracking formation + `stage3.mp4` | Background subtraction, contour shapes, shape-preserving assignment, same dynamics in fixed box |

Use this file to prepare answers about **implementation** (which functions do what), **mathematical models** (equations and meaning), and **design choices** (why we used them).
