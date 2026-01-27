# Illuminated Drone Show Simulation

A Python simulation where a swarm of "drones" (particles) forms shapes in three stages: a handwritten word, a greeting text, then tracking an object from video. Outputs are MP4 animations and trajectory data.

## Stages

1. **Stage 1 — Handwritten word**  
   Loads a handwritten word image, binarizes it, samples target points (contour + fill), and simulates the swarm from a grid to the word shape.

2. **Stage 2 — Greeting text**  
   Loads a second image (e.g. "Happy New Year"), samples targets from it, and simulates the swarm from the Stage 1 formation to the new text.

3. **Stage 3 — Video object tracking**  
   Extracts a moving object from a video (background subtraction), gets shape targets per frame, and simulates the swarm following the object over time. Skipped if no video is provided.

## Requirements

- Python 3.7+
- ffmpeg (for writing MP4 files)

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `Pillow`, `matplotlib`, `opencv-python`.

## Usage

Put the test files in `inputs/` and run:

```bash
python main.py
```

| Test file | Stage | Description |
|-----------|--------|-------------|
| `handwritten.jpeg` | 1 | Handwritten word; binarized, component-filtered, contour+fill sampling. |
| `happy_new_year.png` | 2 | Greeting text; same pipeline as Stage 1. |
| `video.mp4` | 3 | Moving object extracted by background subtraction; swarm tracks shape per frame. |

Outputs go to `outputs/`: `stage1.mp4`, `stage2.mp4`, `stage3.mp4` and `traj_stage1.npy`, `traj_stage2.npy`, `traj_stage3.npy`. If the video is missing, Stage 3 is skipped with a warning.

### Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--word_image` | `inputs/handwritten.jpeg` | Stage 1 image |
| `--text_image` | `inputs/happy_new_year.png` | Stage 2 image |
| `--video` | `inputs/video.mp4` | Stage 3 video (optional) |
| `--outdir` | `outputs` | Output directory |
| `--N` | `400` | Number of drones |
| `--seed` | `7` | Random seed |
| `--dt` | `0.02` | Time step |
| `--steps1` | `1200` | Max steps Stage 1 |
| `--steps2` | `1500` | Max steps Stage 2 |
| `--max_video_frames` | `220` | Max frames from video |

Example: `python main.py --word_image inputs/my_word.jpg --N 300 --outdir my_outputs`

## Input requirements (when it works well)

- **Stage 1:** Word clearly visible, good ink/background contrast, well lit, full word in frame.
- **Stage 2:** Legible text, clear edges, relatively uniform background.
- **Stage 3:** Object moving on a stable background, visually distinct; object not too small or transparent.

## When it may not work (test cases / limitations)

| Stage | Case | Expected issue |
|-------|------|----------------|
| **1** | Cropped word | Missing letters. |
| **1** | Low contrast (pencil, faint ink) | Weak or no mask; binarization fails. |
| **1** | Multiple words, decoration | Extra blobs or wrong letters. |
| **1** | Tiny/huge word in frame | Too few pixels or filter drops parts. |
| **1** | Strong shadows, folds | Extra/missing components; stray point groups. |
| **2** | Gradient or textured background | Noisy or wrong text mask. |
| **2** | Very thin or script font | Gaps in letters after morphology. |
| **2** | Multi-line or crowded layout | Awkward assignment or layout. |
| **2** | Low resolution | Sparse or blocky formation. |
| **3** | Moving or changing background | Wrong background model; noisy/empty mask. |
| **3** | Object similar to background | Subtraction fails; swarm does not follow. |
| **3** | Multiple moving objects | Tracks one only (e.g. largest). |
| **3** | Very small object | Jittery or empty targets. |
| **3** | Object leaves frame | Fallback to previous shape or empty. |

Checklist: word fully visible with clear contrast (Stage 1); single clear text image with simple background (Stage 2); one dominant object on a stable, distinguishable background (Stage 3).

## How it works (short)

- **Target generation:** Images are binarized, morphologically cleaned, and filtered by connected-component area. Points are sampled from contours and filled regions (farthest-point sampling), then normalized and mapped to a world box.
- **Dynamics:** Each drone is driven toward its assigned target (PD control) with pairwise repulsion when closer than a safe distance. Integration is RK4; simulation can stop early when drones are close to targets.
- **Assignment:** Drones are assigned to targets (nearest-neighbor or order-preserving). In Stage 3, assignment is updated per frame to follow the moving shape.
