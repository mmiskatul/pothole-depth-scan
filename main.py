"""
Pothole detection + depth-severity visualization.

Runs a YOLO pothole detector over a video, estimates how *deep* each pothole is
relative to the road surface around it, and renders a colour-coded overlay video
(green = shallow  ->  red = critical).

Depth note
----------
A single camera cannot measure metric depth. What is measured here is the
*depression severity*: the interior of the pothole is compared against a ring of
surrounding road. With the neural backend that comparison happens on a monocular
depth map (Depth Anything V2); with the shadow backend it happens on image
luminance (deep holes self-shadow). Both are relative, scale-free scores in 0..1.

Usage
-----
    python main.py --source road.mp4
    python main.py --source road.mp4 --output out.mp4 --show-depth --display
    python main.py --source 0 --depth-backend shadow          # webcam, no download
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Appearance
# --------------------------------------------------------------------------- #

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Severity colour ramp, BGR. Stops are (position, colour).
RAMP = [
    (0.00, (105, 205, 80)),    # calm green
    (0.35, (60, 215, 235)),    # yellow
    (0.65, (30, 140, 255)),    # orange
    (0.88, (48, 48, 238)),     # red
    (1.00, (26, 26, 178)),     # dark red
]

TIERS = [(0.25, "MINOR"), (0.50, "MODERATE"), (0.75, "SEVERE"), (1.01, "CRITICAL")]

INK = (245, 245, 245)
PANEL = (26, 24, 22)


def build_lut() -> np.ndarray:
    """256-entry BGR lookup table interpolated from RAMP."""
    lut = np.zeros((256, 3), np.uint8)
    for i in range(256):
        t = i / 255.0
        for (p0, c0), (p1, c1) in zip(RAMP, RAMP[1:]):
            if p0 <= t <= p1:
                f = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
                lut[i] = [int(round(a + (b - a) * f)) for a, b in zip(c0, c1)]
                break
        else:
            lut[i] = RAMP[-1][1]
    return lut


LUT = build_lut()


def severity_color(s: float) -> tuple:
    return tuple(int(v) for v in LUT[int(np.clip(s, 0.0, 1.0) * 255)])


def severity_tier(s: float) -> str:
    for limit, name in TIERS:
        if s < limit:
            return name
    return TIERS[-1][1]


# --------------------------------------------------------------------------- #
# Depth / severity field
# --------------------------------------------------------------------------- #

class DepthField:
    """Produces a per-frame float map where HIGHER means 'closer to camera'.

    neural : Depth Anything V2 inverse relative depth.
    shadow : image luminance (no model download; a pothole reads as a dark pit).
    """

    def __init__(self, backend="auto", device="cpu", size=392,
                 model_id="depth-anything/Depth-Anything-V2-Small-hf"):
        self.size = int(size)
        self.device = device
        self.backend = backend
        self.model = None
        self.processor = None

        if backend in ("auto", "neural"):
            try:
                import torch
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                self.torch = torch
                print("[depth] loading %s on %s ..." % (model_id, device), flush=True)
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
                self.model.to(device).eval()
                self.backend = "neural"
                print("[depth] neural backend ready", flush=True)
            except Exception as exc:                        # offline, no deps, OOM ...
                if backend == "neural":
                    raise
                print("[depth] neural backend unavailable (%s: %s)"
                      % (exc.__class__.__name__, exc))
                print("[depth] falling back to 'shadow' backend")
                self.backend = "shadow"
        else:
            self.backend = "shadow"

    # -- inference ---------------------------------------------------------- #

    def __call__(self, frame):
        if self.backend == "neural":
            return self._neural(frame)
        return self._shadow(frame)

    def _neural(self, frame):
        h, w = frame.shape[:2]
        scale = self.size / max(h, w)
        if scale < 1:
            small = cv2.resize(frame, (max(32, int(w * scale)), max(32, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            pred = self.model(**inputs).predicted_depth      # (1, h', w') inverse depth

        depth = pred.squeeze().float().cpu().numpy()
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        return _normalize(depth)

    @staticmethod
    def _shadow(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        lab = cv2.bilateralFilter(lab, 7, 40, 7)
        return _normalize(lab)


def _normalize(a):
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(a, np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def measure_depression(field_map, box, sensitivity=0.075, mode="signed"):
    """Compare pothole interior against the ring of road around it.

    mode="signed"    : only interiors that read *further/darker* than the road
                       count as deep. Correct for dashcam / oblique views.
    mode="magnitude" : any strong deviation from the road counts, in either
                       direction. Use for top-down aerial footage, where a pit
                       may be filled with pale silt or dark water and the sign
                       of the contrast says nothing about how developed it is.

    Returns (severity 0..1, raw contrast in units of local road roughness).
    """
    H, W = field_map.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return 0.0, 0.0

    # Interior: centred ellipse, keeps rim pixels out of the sample.
    core = np.zeros((H, W), np.uint8)
    cv2.ellipse(core, ((x1 + x2) // 2, (y1 + y2) // 2),
                (max(2, int(bw * 0.33)), max(2, int(bh * 0.33))), 0, 0, 360, 255, -1)

    # Reference: road to the LEFT and RIGHT of the box, over the same image
    # rows. In a forward-facing view depth changes mostly with row -- the road
    # recedes toward the horizon -- so a full surrounding ring would sample
    # far-away tarmac above the box and wreck the comparison. Lateral
    # neighbours sit at the same distance from the camera.
    gap = max(2, int(0.12 * bw))
    reach = max(6, int(0.60 * bw))
    outer = np.zeros((H, W), np.uint8)
    cv2.rectangle(outer, (x1 - gap - reach, y1), (x1 - gap, y2), 255, -1)
    cv2.rectangle(outer, (x2 + gap, y1), (x2 + gap + reach, y2), 255, -1)

    if int(outer.sum()) // 255 < 24:      # box hugs a frame edge: full ring
        pad_in = int(0.10 * max(bw, bh))
        pad_out = int(0.55 * max(bw, bh))
        outer[:] = 0
        cv2.rectangle(outer, (x1 - pad_out, y1 - pad_out),
                      (x2 + pad_out, y2 + pad_out), 255, -1)
        cv2.rectangle(outer, (x1 - pad_in, y1 - pad_in),
                      (x2 + pad_in, y2 + pad_in), 0, -1)

    ring_ys, ring_xs = np.nonzero(outer)
    core_ys, core_xs = np.nonzero(core)
    if core_ys.size < 12 or ring_ys.size < 40:
        return 0.0, 0.0

    # Fit the local road surface as depth = a*row + b over the reference pixels.
    # In a forward-facing view the road recedes with image row, so this slope
    # IS the road; a pothole shows up as a residual below it. On a top-down
    # view the slope is ~0 and the fit degrades gracefully to a flat mean.
    ring_v = field_map[ring_ys, ring_xs]
    A = np.stack([ring_ys.astype(np.float32), np.ones(ring_ys.size, np.float32)], 1)
    coef, _, _, _ = np.linalg.lstsq(A, ring_v, rcond=None)

    # Road roughness = spread of the reference pixels about that fitted plane.
    # Grading against it makes the score a scale-free "how many times rougher
    # than the surrounding road is this dip", valid near and far from camera.
    ring_resid = ring_v - (A @ coef)
    sigma = 1.4826 * float(np.median(np.abs(ring_resid - np.median(ring_resid))))
    if sigma < 1e-5:
        return 0.0, 0.0

    core_pred = coef[0] * core_ys.astype(np.float32) + coef[1]
    core_resid = float(np.median(core_pred - field_map[core_ys, core_xs]))

    contrast = core_resid / sigma          # positive = interior sits below road
    graded = abs(contrast) if mode == "magnitude" else contrast
    sev = float(np.clip(graded / max(sensitivity, 1e-3), 0.0, 1.0))
    return sev, contrast


# --------------------------------------------------------------------------- #
# Lightweight IoU tracker (stable ids + temporal smoothing of severity)
# --------------------------------------------------------------------------- #

@dataclass
class Track:
    tid: int
    box: np.ndarray
    conf: float
    severity: float
    raw: float
    age: int = 0
    misses: int = 0
    hits: int = 1
    history: list = field(default_factory=list)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / max(ua, 1e-6))


class Tracker:
    def __init__(self, iou_thr=0.25, max_age=12, smooth=0.35):
        self.iou_thr, self.max_age, self.smooth = iou_thr, max_age, smooth
        self.tracks = []
        self._next = 1

    def update(self, dets):
        for t in self.tracks:
            t.age += 1
            t.misses += 1

        used = set()
        for box, conf, sev, raw in dets:
            best, best_i = None, self.iou_thr
            for t in self.tracks:
                if id(t) in used:
                    continue
                v = iou(t.box, box)
                if v >= best_i:
                    best, best_i = t, v
            if best is None:
                best = Track(self._next, np.asarray(box, np.float32), conf, sev, raw)
                self._next += 1
                self.tracks.append(best)
            else:
                a = self.smooth
                best.box = (1 - a) * best.box + a * np.asarray(box, np.float32)
                best.conf = (1 - a) * best.conf + a * conf
                best.severity = (1 - a) * best.severity + a * sev
                best.raw = (1 - a) * best.raw + a * raw
                best.hits += 1
            best.misses = 0
            best.history.append(best.severity)
            if len(best.history) > 90:
                best.history.pop(0)
            used.add(id(best))

        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return [t for t in self.tracks if t.misses == 0]


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #

def blend(dst, overlay, alpha, rect=None):
    if rect is None:
        cv2.addWeighted(overlay, alpha, dst, 1 - alpha, 0, dst)
        return dst
    x1, y1, x2, y2 = rect
    roi, ov = dst[y1:y2, x1:x2], overlay[y1:y2, x1:x2]
    if roi.size:
        cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0, roi)
    return dst


def rounded_rect(img, p1, p2, color, radius=8, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy, a0 in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
                           (x1 + r, y2 - r, 90), (x2 - r, y2 - r, 0)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, color, -1)
    else:
        cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)


def corner_box(img, box, color, thickness=2, frac=0.24):
    x1, y1, x2, y2 = [int(v) for v in box]
    lx = max(6, int((x2 - x1) * frac))
    ly = max(6, int((y2 - y1) * frac))
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * lx, py), color, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * ly), color, thickness, cv2.LINE_AA)


def text(img, s, org, scale=0.5, color=INK, thick=1, shadow=True):
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0),
                    thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_pothole(frame, t, scale, occupied=None):
    x1, y1, x2, y2 = [int(v) for v in t.box]
    color = severity_color(t.severity)
    tier = severity_tier(t.severity)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax, ay = max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)

    # Translucent fill following the pothole's elliptical footprint.
    ov = frame.copy()
    cv2.ellipse(ov, (cx, cy), (ax, ay), 0, 0, 360, color, -1)
    pad = 4
    blend(frame, ov, 0.10 + 0.22 * t.severity,
          (max(0, x1 - pad), max(0, y1 - pad),
           min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)))

    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, color,
                max(1, int(1.5 * scale)), cv2.LINE_AA)
    corner_box(frame, t.box, color, max(2, int(2 * scale)))

    # Label chip.
    label = "#%d  %s" % (t.tid, tier)
    meta = "depth %.2f   conf %d%%" % (t.severity, round(t.conf * 100))
    fs, fs2 = 0.52 * scale, 0.42 * scale
    (tw, _), _ = cv2.getTextSize(label, FONT, fs, 1)
    (mw, _), _ = cv2.getTextSize(meta, FONT, fs2, 1)
    cw = int(max(tw, mw) + 18 * scale)
    ch = int(38 * scale)
    cx1 = max(2, min(x1, frame.shape[1] - cw - 2))
    cy1 = y1 - ch - int(6 * scale)
    above = True
    if cy1 < 2:
        cy1 = min(frame.shape[0] - ch - 2, y2 + int(6 * scale))
        above = False
    # Nudge the chip until it stops colliding with chips already placed this
    # frame, so clustered potholes stay readable.
    if occupied:
        step = ch + int(3 * scale)
        for attempt in range(14):
            rect = (cx1, cy1, cx1 + cw, cy1 + ch)
            if not any(_overlaps(rect, o) for o in occupied):
                break
            direction = -1 if above else 1
            nxt = cy1 + direction * step * (1 if attempt % 2 == 0 else -1) * (attempt // 2 + 1)
            if 2 <= nxt <= frame.shape[0] - ch - 2:
                cy1 = nxt
    cy2, cx2 = cy1 + ch, cx1 + cw
    if occupied is not None:
        occupied.append((cx1, cy1, cx2, cy2))

    chip = frame.copy()
    rounded_rect(chip, (cx1, cy1), (cx2, cy2), PANEL, int(7 * scale))
    blend(frame, chip, 0.78, (cx1, cy1, cx2, cy2))
    rounded_rect(frame, (cx1, cy1), (cx2, cy2), color, int(7 * scale), 1)
    cv2.rectangle(frame, (cx1, cy1), (cx1 + max(2, int(4 * scale)), cy2), color, -1)

    text(frame, label, (cx1 + int(10 * scale), cy1 + int(15 * scale)), fs, color, 1, False)
    text(frame, meta, (cx1 + int(10 * scale), cy1 + int(30 * scale)), fs2,
         (205, 205, 205), 1, False)

    # Leader line from chip to pothole.
    cv2.line(frame, ((cx1 + cx2) // 2, cy2 if above else cy1),
             (cx, y1 if above else y2), color, 1, cv2.LINE_AA)


def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_hud(frame, stats, fps, frame_i, total, backend, scale):
    pw, ph = int(250 * scale), int(112 * scale)
    m = int(14 * scale)
    ov = frame.copy()
    rounded_rect(ov, (m, m), (m + pw, m + ph), PANEL, int(10 * scale))
    blend(frame, ov, 0.72, (m, m, m + pw, m + ph))
    rounded_rect(frame, (m, m), (m + pw, m + ph), (70, 66, 62), int(10 * scale), 1)

    text(frame, "POTHOLE  DEPTH  SCAN", (m + int(12 * scale), m + int(20 * scale)),
         0.48 * scale, (250, 250, 250), 1, False)
    cv2.line(frame, (m + int(12 * scale), m + int(27 * scale)),
             (m + pw - int(12 * scale), m + int(27 * scale)), (70, 66, 62), 1)

    prog = ("%d/%d" % (frame_i, total)) if total else str(frame_i)
    text(frame, "frame " + prog, (m + int(12 * scale), m + int(43 * scale)),
         0.4 * scale, (185, 185, 185), 1, False)
    text(frame, "%5.1f fps   %s" % (fps, backend),
         (m + int(12 * scale), m + int(58 * scale)), 0.4 * scale, (185, 185, 185), 1, False)

    y = m + int(78 * scale)
    x = m + int(12 * scale)
    for limit, name in TIERS:
        c = severity_color(max(0.0, limit - 0.13))
        cv2.circle(frame, (x + int(4 * scale), y - int(4 * scale)),
                   int(4 * scale), c, -1, cv2.LINE_AA)
        text(frame, str(stats.get(name, 0)), (x + int(12 * scale), y),
             0.42 * scale, c, 1, False)
        x += int(58 * scale)
    text(frame, "minor    mod     sev     crit",
         (m + int(12 * scale), y + int(15 * scale)), 0.33 * scale, (150, 150, 150), 1, False)


def draw_legend(frame, scale):
    h = frame.shape[0]
    bw, bh = int(210 * scale), int(12 * scale)
    x1 = int(14 * scale)
    y1 = h - bh - max(40, int(46 * scale))
    grad = np.zeros((bh, bw, 3), np.uint8)
    for i in range(bw):
        grad[:, i] = severity_color(i / max(1, bw - 1))
    frame[y1:y1 + bh, x1:x1 + bw] = grad
    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (70, 66, 62), 1)
    fs = max(0.36, 0.36 * scale)
    gap = max(13, int(14 * scale))
    text(frame, "shallow", (x1, y1 - gap), fs, (180, 180, 180), 1)
    (tw, _), _ = cv2.getTextSize("deep", FONT, fs, 1)
    text(frame, "deep", (x1 + bw - tw, y1 - gap), fs, (180, 180, 180), 1)
    text(frame, "relative depression depth", (x1, y1 + bh + max(21, int(23 * scale))),
         max(0.34, 0.34 * scale), (140, 140, 140), 1)


def draw_depth_pip(frame, field_map, scale):
    h, w = frame.shape[:2]
    pw = max(64, int(w * 0.24))
    ph = max(48, int(pw * h / w))
    small = cv2.resize((field_map * 255).astype(np.uint8), (pw, ph),
                       interpolation=cv2.INTER_AREA)
    cmap = cv2.applyColorMap(small, cv2.COLORMAP_INFERNO)
    x1, y1 = w - pw - int(14 * scale), int(14 * scale)
    frame[y1:y1 + ph, x1:x1 + pw] = cmap
    cv2.rectangle(frame, (x1, y1), (x1 + pw, y1 + ph), (70, 66, 62), 1)
    text(frame, "depth", (x1 + int(6 * scale), y1 + int(14 * scale)), 0.38 * scale, INK, 1)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def open_writer(path, fps, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    for tag in ("mp4v", "avc1", "XVID"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*tag), fps, size)
        if w.isOpened():
            return w
        w.release()
    raise RuntimeError("cannot open a video writer for %s" % path)


def run(args):
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        print("error: weights not found: %s" % weights, file=sys.stderr)
        return 2

    source = int(args.source) if str(args.source).isdigit() else str(args.source)
    if isinstance(source, str) and not Path(source).exists():
        print("error: source not found: %s" % source, file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("error: cannot open source: %s" % source, file=sys.stderr)
        return 2

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Order matters: drop the skipped head first, then apply the cap.
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        total = max(0, total - args.start_frame)
        print("[video] skipping first %d frames" % args.start_frame)
    if args.max_frames:
        total = min(total, args.max_frames) if total else args.max_frames

    print("[video] %dx%d @ %.2f fps, %s frames" % (W, H, fps, total or "?"))
    print("[model] loading %s" % weights)
    model = YOLO(str(weights))

    depth = DepthField(args.depth_backend, args.device, args.depth_size)

    # Luminance contrast is far stronger than depth-map contrast, so the two
    # backends need different grading scales.
    sensitivity = args.sensitivity
    if sensitivity is None:
        sensitivity = 4.0 if args.contrast_mode == "magnitude" else 3.0
    print("[grade] sensitivity %.3f  mode %s  (%s backend)"
          % (sensitivity, args.contrast_mode, depth.backend))
    tracker = Tracker(smooth=args.smooth)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("outputs") / (Path(str(source)).stem + "_depth.mp4")
    writer = open_writer(out_path, fps, (W, H))
    scale = max(0.90, min(2.0, W / 1280.0))

    roi = None
    if args.roi:
        try:
            roi = [float(v) for v in args.roi.split(",")]
            assert len(roi) == 4 and roi[0] < roi[2] and roi[1] < roi[3]
        except Exception:
            print("error: --roi wants x1,y1,x2,y2 as fractions, e.g. 0.05,0.2,0.95,0.62",
                  file=sys.stderr)
            return 2
        print("[roi]   keeping detections centred inside %s" % roi)

    rows = []
    field_map = np.zeros((H, W), np.float32)
    frame_i = 0
    t0 = time.time()
    ema_fps = 0.0

    try:
        from tqdm import tqdm
        bar = tqdm(total=total or None, unit="f", ncols=88)
    except Exception:
        bar = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            if args.max_frames and frame_i > args.max_frames:
                frame_i -= 1
                break
            tick = time.time()

            if (frame_i - 1) % max(1, args.depth_stride) == 0:
                field_map = depth(frame)

            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                device=args.device, verbose=False)[0]

            dets = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for box, cf in zip(xyxy, confs):
                    if roi is not None:
                        mx = (box[0] + box[2]) / 2.0 / W
                        my = (box[1] + box[3]) / 2.0 / H
                        if not (roi[0] <= mx <= roi[2] and roi[1] <= my <= roi[3]):
                            continue
                    sev, raw = measure_depression(field_map, box, sensitivity, args.contrast_mode)
                    dets.append((box, float(cf), sev, raw))

            tracks = tracker.update(dets)

            stats = dict((name, 0) for _, name in TIERS)
            hud_w, hud_h = int(250 * scale), int(112 * scale)
            m = int(14 * scale)
            occupied = [(0, 0, m + hud_w, m + hud_h),
                        (0, H - int(80 * scale), int(240 * scale), H)]
            for t in sorted(tracks, key=lambda t: -t.severity):
                if t.hits < args.min_hits:
                    continue
                stats[severity_tier(t.severity)] += 1
                draw_pothole(frame, t, scale, occupied)
                if args.csv:
                    rows.append([frame_i, t.tid] +
                                [round(float(v), 1) for v in t.box] +
                                [round(t.conf, 3), round(t.severity, 3),
                                 round(t.raw, 4), severity_tier(t.severity)])

            if args.show_depth:
                draw_depth_pip(frame, field_map, scale)

            inst = 1.0 / max(1e-6, time.time() - tick)
            ema_fps = inst if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst
            draw_hud(frame, stats, ema_fps, frame_i, total, depth.backend, scale)
            draw_legend(frame, scale)

            writer.write(frame)
            if bar is not None:
                bar.update(1)
                bar.set_postfix_str("%d potholes  %.1ffps" % (sum(stats.values()), ema_fps))
            elif frame_i % 25 == 0:
                print("  frame %d/%s  %.1f fps" % (frame_i, total or "?", ema_fps))

            if args.display:
                view = frame if W <= 1280 else cv2.resize(frame, (1280, int(1280 * H / W)))
                cv2.imshow("pothole depth scan  [q to quit]", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("\n[stopped by user]")
                    break
    finally:
        if bar is not None:
            bar.close()
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "id", "x1", "y1", "x2", "y2", "conf",
                        "severity", "raw_contrast", "tier"])
            w.writerows(rows)
        print("[csv]   %s  (%d rows)" % (args.csv, len(rows)))

    dt = time.time() - t0
    uniq = dict((t.tid, severity_tier(t.severity))
                for t in tracker.tracks if t.hits >= args.min_hits)
    print("\n[done]  %d frames in %.1fs (%.1f fps)" % (frame_i, dt, frame_i / max(dt, 1e-6)))
    if uniq:
        print("[last]  %d tracked potholes still open at end of clip" % len(uniq))
    print("[out]   %s" % out_path.resolve())
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Detect potholes in a video and colour them by relative depth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", required=True, help="video file path, or a webcam index like 0")
    p.add_argument("--weights", default="best.pt", help="YOLO pothole weights")
    p.add_argument("--output", default=None, help="output mp4 (default outputs/<name>_depth.mp4)")

    p.add_argument("--conf", type=float, default=0.30, help="detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.50, help="detection NMS IoU")
    p.add_argument("--imgsz", type=int, default=640, help="detector input size")
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0 ...")

    p.add_argument("--depth-backend", choices=["auto", "neural", "shadow"], default="auto",
                   help="auto = neural if available, else shadow")
    p.add_argument("--depth-size", type=int, default=392, help="long edge fed to the depth model")
    p.add_argument("--depth-stride", type=int, default=1,
                   help="recompute depth every N frames (2-3 speeds up CPU runs a lot)")
    p.add_argument("--sensitivity", type=float, default=None,
                   help="road-roughness multiples mapping to severity 1.0; lower = "
                        "harsher grading (default: 3.0 signed, 4.0 magnitude)")

    p.add_argument("--contrast-mode", choices=["signed", "magnitude"], default="signed",
                   help="signed = only darker/further interiors count as deep (dashcam); "
                        "magnitude = any strong deviation counts (top-down aerial)")

    p.add_argument("--smooth", type=float, default=0.35, help="temporal smoothing factor (0-1)")
    p.add_argument("--min-hits", type=int, default=2, help="frames before a track is drawn")

    p.add_argument("--show-depth", action="store_true", help="depth map picture-in-picture")
    p.add_argument("--display", action="store_true", help="live preview window")
    p.add_argument("--roi", default=None,
                   help="keep only detections centred in this box, as fractions "
                        "x1,y1,x2,y2 -- use it to mask out camera-rig hardware "
                        "(e.g. 0.05,0.2,0.95,0.62 for a handlebar-mounted camera)")
    p.add_argument("--start-frame", type=int, default=0,
                   help="skip this many frames before processing (e.g. past a black lead-in)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--csv", default=None, help="write per-frame measurements to this csv")
    return p.parse_args(argv)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    sys.exit(run(parse_args()))
