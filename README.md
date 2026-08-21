# Pothole Depth Scan

Detects potholes in road video with YOLO and grades each one by how **deep** it
sits relative to the surrounding road, rendering a colour-coded overlay video
(green = shallow → red = critical).

![tiers](https://img.shields.io/badge/tiers-MINOR%20%7C%20MODERATE%20%7C%20SEVERE%20%7C%20CRITICAL-informational)

## What it produces

- An annotated MP4: translucent elliptical fill + corner brackets per pothole,
  coloured by severity, with a label chip carrying a stable track ID, severity
  tier, depth score and confidence.
- A HUD with frame counter, live FPS, active backend and per-tier counts, plus
  a gradient legend and an optional depth-map picture-in-picture.
- An optional CSV of every per-frame measurement for downstream analysis.

Label chips nudge themselves apart so clustered potholes stay readable, and an
IoU tracker smooths severity over time so colours don't flicker.

## How the depth measurement works

A single camera cannot measure metric depth. What this measures is **relative
depression severity**: the pothole interior is compared against the road
immediately beside it.

1. **Sample the interior** — a centred ellipse, which keeps rim pixels out.
2. **Sample the reference** — road to the *left and right* of the box, over the
   same image rows. A full surrounding ring would sample distant tarmac above
   the box; in a forward-facing view the road recedes toward the horizon, so
   that reference is far further away and the comparison collapses.
3. **Fit the local road plane** — regress depth against image row across those
   reference pixels. In a forward view that slope *is* the road; a pothole
   shows up as a residual below it. On a top-down view the slope is ~0 and the
   fit degrades gracefully to a flat mean.
4. **Normalise by road roughness** — divide the interior's residual by the
   spread of the reference pixels about the fitted plane. The score becomes a
   scale-free "how many times rougher than the surrounding road is this dip",
   valid near and far from the camera alike.

### Two fields to measure on

| Backend | Signal | Use when |
|---|---|---|
| `neural` | Depth Anything V2 inverse relative depth | oblique / forward views with real depth relief |
| `shadow` | image luminance (a pit self-shadows) | aerial or flat views, or offline — no download |

### Two grading modes

| Mode | Counts as deep | Use when |
|---|---|---|
| `signed` | only interiors reading further/darker than the road | dashcam, oblique views |
| `magnitude` | any strong deviation, either direction | top-down aerial, where a pit may hold pale silt or dark water and the sign says nothing |

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
```

Place your trained YOLO pothole weights at `best.pt` (not in this repo — see
*Not included* below).

## Usage

```bash
# defaults: neural depth, signed grading
python main.py --source road.mp4

# forward-facing / dashcam, masking out camera-rig hardware
python main.py --source ride.mp4 --imgsz 960 --conf 0.15 \
    --roi 0.15,0.22,0.85,0.55 --start-frame 1240

# top-down aerial footage
python main.py --source drone.mp4 --depth-backend shadow \
    --contrast-mode magnitude --imgsz 960 --conf 0.18

# live preview + depth PiP + measurements
python main.py --source road.mp4 --display --show-depth --csv out.csv
```

### Options worth knowing

| Flag | Purpose |
|---|---|
| `--roi x1,y1,x2,y2` | keep only detections centred in this fractional box — masks out mirrors, handlebars, bonnet |
| `--start-frame N` | skip a black lead-in or intro |
| `--depth-stride N` | recompute depth every N frames; 2–3 speeds up CPU runs a lot |
| `--sensitivity` | road-roughness multiples mapping to severity 1.0 (default 3.0 signed / 4.0 magnitude) |
| `--device cuda` | GPU inference |

Note that the pipeline writes `mp4v`, which browsers and most social platforms
won't decode. Re-encode to H.264 before uploading anywhere:

```bash
ffmpeg -i outputs/road_depth.mp4 -c:v libx264 -profile:v high -pix_fmt yuv420p \
       -crf 20 -movflags +faststart outputs/road_depth_h264.mp4
```

## Performance

Roughly 1.3–1.8 fps at 1080p on CPU (Python 3.13, torch CPU). The neural depth
backend costs ~0.5 s/frame at 392 px. Use `--depth-stride`, `--depth-backend
shadow`, or a GPU to speed things up.

## Known limits

These are measured, not hypothetical:

- **Monocular depth cannot resolve pothole relief at road distance.** On
  motorcycle-mounted footage the neural backend produced a median residual of
  −0.58 with only 27% in the physically correct direction, and raising the
  depth model to 728 px made it *worse*. Pothole depth is centimetres at road
  distance — below what a relative depth model resolves. On such footage the
  `shadow` field is the honest signal (median +1.08, 80% correct direction),
  but it grades **surface contrast, not geometry**.
- **`magnitude` mode cannot distinguish a deep pit from a shallow patch of
  contrasting fill.** It measures deviation strength only.
- **Detector transfer is the real bottleneck.** A model trained on discrete
  potholes performs poorly on uniformly eroded asphalt, and raising `--conf`
  does not fix it — roadside objects (concrete blocks, mirrors) can score
  *higher* than genuine potholes, so confidence tuning removes real detections
  first. `--roi` handles fixed rig hardware; misclassified objects need
  retraining.

Treat the depth scores as a **relative triage ranking**, not as measured depth.

## Not included

Excluded via `.gitignore`:

- `best.pt` — the trained YOLO pothole checkpoint (~44 MB)
- `*.mp4` and `outputs/` — source footage and rendered results
- `.venv/` — the virtual environment

## Layout

```
main.py               detection, depth grading, rendering, tracking
requirements.txt
```
