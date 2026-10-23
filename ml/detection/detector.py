"""
detector.py — Multi-Model YOLOv8 Ensemble Detector (v5 — Strong COCO Authority + SAHI Sliced Inference)
========================================================================================================
Thread-safe singleton that runs YOLOv8 ensemble with:
  - yolov8m.pt as the COCO authority for car/truck/bus/bicycle/motorcycle/person
  - Custom trained models (aerial two_wheeler, ground bike/scooter, cycles) for specialized recall
  - SAHI-style sliced inference (640px tiles, 25% overlap) for dense/high-res parking lots
  - Strict box sanity filtering (rejects strips and huge/tiny boxes)
  - Containment-aware NMS deduplication (eliminates ghost sub-boxes)
  - Separate person detection for rider suppression

Usage:
    detector = TwoWheelerDetector.get_instance()
    detections, persons = detector.detect_frame_with_persons(frame, conf=0.30)
    detections = detector.detect_frame(frame, conf=0.30)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

# ── Lazy imports ──────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None  # type: ignore

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ── Locate project root ────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    """Walk up from this file until we find config/pipeline.yaml."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config" / "pipeline.yaml").exists():
            return parent
    return here.parents[2]


PROJECT_ROOT = _find_project_root()


# ── Config loader ──────────────────────────────────────────────────────────────

def _load_pipeline_config() -> dict:
    """Load config/pipeline.yaml; fall back to sensible defaults."""
    config_path_env = os.environ.get("PIPELINE_CONFIG_PATH", "")
    candidates = [
        Path(config_path_env) if config_path_env else None,
        PROJECT_ROOT / "config" / "pipeline.yaml",
        Path("config/pipeline.yaml"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            if YAML_AVAILABLE:
                with open(candidate) as f:
                    return yaml.safe_load(f) or {}
    return {}


# ── Detection dataclass ────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single object detection from YOLO inference."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (absolute pixels)
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def to_tlwh(self) -> Tuple[float, float, float, float]:
        """Convert to (top, left, width, height) — used by DeepSORT."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)

    def to_xywh(self) -> List[float]:
        """[x_center, y_center, width, height] in absolute pixels."""
        cx, cy = self.center
        return [cx, cy, self.width, self.height]

    def to_dict(self) -> dict:
        return {
            "bbox": [round(float(v), 2) for v in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "confidence_percent": round(float(self.confidence) * 100, 1),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center": [round(float(v), 2) for v in self.center],
        }


# ── Canonical class sets ───────────────────────────────────────────────────────
TWO_WHEELER_NAMES = {"motorcycle", "bike", "scooter", "motorbike", "two_wheeler",
                     "two-wheeler", "moped", "scooty"}
BICYCLE_NAMES = {"bicycle", "cycle", "pushbike"}
COCO_VEHICLE_NAMES = {"motorcycle", "bicycle", "car", "truck", "bus"}
COCO_PERSON_NAME = "person"
ALL_VEHICLE_NAMES = TWO_WHEELER_NAMES | BICYCLE_NAMES | {"car", "truck", "bus"}


# ── Box sanity filter ──────────────────────────────────────────────────────────

def _box_is_valid(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
    class_name: str,
    verbose: bool = False,
) -> bool:
    """
    Return True only if the box is geometrically sane.
    Rejects:
      - Tiny boxes (< 20px)
      - Extreme aspect ratios (w/h > 4.5 or < 0.22)
      - Two-wheelers covering > 22% of image area, or >70% image dimension
      - Cars/trucks/buses covering > 28% of image area
    """
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        if verbose:
            print(f"[BOX_FILTER] DROP zero-size: {class_name} ({w:.0f}x{h:.0f})")
        return False

    if w < 20 or h < 20:
        if verbose:
            print(f"[BOX_FILTER] DROP tiny: {class_name} ({w:.0f}x{h:.0f})")
        return False

    aspect = w / max(h, 1)
    if aspect > 4.5:
        if verbose:
            print(f"[BOX_FILTER] DROP horizontal strip: {class_name} aspect={aspect:.2f}")
        return False
    if aspect < 0.22:
        if verbose:
            print(f"[BOX_FILTER] DROP vertical strip: {class_name} aspect={aspect:.2f}")
        return False

    area_frac = (w * h) / max(img_w * img_h, 1)
    is_tw = class_name.lower() in (TWO_WHEELER_NAMES | BICYCLE_NAMES)

    if is_tw:
        if area_frac > 0.22:
            if verbose:
                print(f"[BOX_FILTER] DROP huge two-wheeler: {class_name} area={area_frac:.2%}")
            return False
        if h > img_h * 0.70:
            if verbose:
                print(f"[BOX_FILTER] DROP too-tall two-wheeler: {class_name} h={h:.0f}/{img_h}")
            return False
        if w > img_w * 0.70:
            if verbose:
                print(f"[BOX_FILTER] DROP too-wide two-wheeler: {class_name} w={w:.0f}/{img_w}")
            return False
    else:
        if area_frac > 0.28:
            if verbose:
                print(f"[BOX_FILTER] DROP huge vehicle: {class_name} area={area_frac:.2%}")
            return False
        # Four-wheelers (cars/trucks/buses) are wide vehicles — cannot have aspect ratio < 0.40
        if aspect < 0.40:
            if verbose:
                print(f"[BOX_FILTER] DROP sliver vehicle: {class_name} aspect={aspect:.2f}")
            return False

    return True


def _clamp_box(x1, y1, x2, y2, img_w, img_h):
    """Clamp bounding box coordinates to image bounds."""
    x1 = max(0.0, min(float(x1), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y2 = max(0.0, min(float(y2), float(img_h)))
    return x1, y1, x2, y2


def _canonical_name(raw_name: str) -> str:
    """Map raw model class name to a canonical class name."""
    n = raw_name.lower().strip()
    if n in TWO_WHEELER_NAMES:
        return "motorcycle"
    if n in BICYCLE_NAMES:
        return "bicycle"
    if n == "car":
        return "car"
    if n in ("truck", "lorry"):
        return "truck"
    if n == "bus":
        return "bus"
    if n == "person":
        return "person"
    return n


# ── NMS & Containment helper ───────────────────────────────────────────────────

def _iou(b1, b2) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(1.0, (b1[2]-b1[0]) * (b1[3]-b1[1]))
    a2 = max(1.0, (b2[2]-b2[0]) * (b2[3]-b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _containment(b1, b2) -> float:
    """Returns fraction of b1 that is enclosed within b2."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(1.0, (b1[2]-b1[0]) * (b1[3]-b1[1]))
    return inter / a1


def _nms(detections: List[Detection], same_class_iou=0.55, cross_class_iou=0.40) -> List[Detection]:
    """
    Weighted NMS with Containment Suppression:
    Sort by confidence descending, suppress duplicates and ghost sub-boxes.
    """
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    for cand in dets:
        is_dup = False
        for k in kept:
            iou_val = _iou(cand.bbox, k.bbox)
            c_cand = _containment(cand.bbox, k.bbox)  # cand inside kept
            c_k = _containment(k.bbox, cand.bbox)     # kept inside cand

            if cand.class_name == k.class_name:
                # 1. Standard IoU suppression
                if iou_val > same_class_iou:
                    is_dup = True
                    break
                # 2. Containment rule: if a smaller box is >85% inside a larger box and IoU > 0.30
                if (c_cand > 0.85 or c_k > 0.85) and iou_val > 0.30:
                    is_dup = True
                    break
                # 3. If candidate is almost completely inside kept (>85%), drop it
                if c_cand > 0.85:
                    is_dup = True
                    break
            else:
                # Cross-class suppression
                if iou_val > cross_class_iou:
                    is_dup = True
                    break
                # If candidate is largely inside an already kept vehicle of another class
                if c_cand > 0.70 or c_k > 0.85:
                    is_dup = True
                    break

        if not is_dup:
            kept.append(cand)
    return kept


class TwoWheelerDetector:
    """
    Thread-safe YOLOv8 multi-model ensemble detector for parking analysis.

    Model roles:
      - yolov8m.pt (COCO): AUTHORITY for car/truck/bus/bicycle/motorcycle/person
      - two_wheeler_detector.pt: custom aerial two-wheeler boost
      - best_twowheller.pt: custom ground-level bike/scooter boost
      - best_cycles.pt: bicycle specialist boost
    """

    _instance: Optional["TwoWheelerDetector"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._models: Dict[str, object] = {}
        self._inference_lock = threading.Lock()
        self._config = _load_pipeline_config()
        self._det_cfg = self._config.get("detection", {})
        self._debug = bool(os.environ.get("DEBUG_DETECT", ""))

        # Performance counters
        self._frame_count = 0
        self._total_inference_ms = 0.0
        self._fps_start = time.time()
        self._last_fps = 0.0

        self._load_all_models()

    @classmethod
    def get_instance(cls) -> "TwoWheelerDetector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_model(self, name: str, path: Path) -> Optional[object]:
        """Load a single YOLO model, return None on failure."""
        if not ULTRALYTICS_AVAILABLE:
            return None
        if not path.exists():
            # If yolov8m.pt is requested and not found locally, let ultralytics download
            if "yolov8m" in str(path) or "yolov8x" in str(path):
                try:
                    model = YOLO(path.name)
                    print(f"[Detector] Downloaded & loaded {name}: {path.name}")
                    return model
                except Exception as e:
                    print(f"[Detector] Could not download {path.name}: {e}")
            print(f"[Detector] SKIP {name}: {path} not found")
            return None
        try:
            model = YOLO(str(path))
            print(f"[Detector] Loaded {name}: {path.name} | classes={list(model.names.values())[:10]}")
            return model
        except Exception as e:
            print(f"[Detector] ERROR loading {name}: {e}")
            return None

    def _load_all_models(self) -> None:
        """Load all ensemble models, using yolov8m.pt as the COCO authority."""
        if not ULTRALYTICS_AVAILABLE:
            print("[Detector] WARNING: ultralytics not installed — no detections will run.")
            return

        # 1. Primary COCO authority model: yolov8m.pt (TASK 1)
        coco_path = PROJECT_ROOT / "yolov8m.pt"
        if not coco_path.exists():
            coco_path = Path("yolov8m.pt")
        m = self._load_model("coco", coco_path)
        if not m:
            # Fallback to yolov8n if yolov8m cannot be loaded
            fallback = PROJECT_ROOT / "backend" / "yolov8n.pt"
            m = self._load_model("coco", fallback)
        if m:
            self._models["coco"] = m

        # 2. Custom aerial two-wheeler detector
        m = self._load_model("aerial", PROJECT_ROOT / "data" / "trained_models" / "two_wheeler_detector.pt")
        if m:
            self._models["aerial"] = m

        # 3. Ground-level bike/scooter specialist
        m = self._load_model("ground", PROJECT_ROOT / "data" / "trained_models" / "best_twowheller.pt")
        if m:
            self._models["ground"] = m

        # 4. Bicycle specialist
        m = self._load_model("cycles", PROJECT_ROOT / "data" / "trained_models" / "best_cycles.pt")
        if m:
            self._models["cycles"] = m

    @property
    def _model_path(self) -> str:
        """Return primary model path for compatibility."""
        if "coco" in self._models:
            m = self._models["coco"]
            return getattr(m, "ckpt_path", "yolov8m.pt")
        return "not loaded"

    @property
    def is_ready(self) -> bool:
        return len(self._models) > 0

    # ── Sliced (SAHI-style) Inference ──────────────────────────────────────────

    def _sliced_inference(
        self,
        frame: np.ndarray,
        conf: float,
    ) -> List[Detection]:
        """
        SAHI-style sliced inference with 640px tiles and 25% overlap.
        Runs yolov8m on tiles to boost recall of small/distant motorcycles.
        """
        if "coco" not in self._models:
            return []

        coco_model = self._models["coco"]
        img_h, img_w = frame.shape[:2]
        tile_size = 640
        stride = int(tile_size * 0.75)  # 25% overlap

        y_starts = list(range(0, max(1, img_h - tile_size + 1), stride))
        if not y_starts or y_starts[-1] + tile_size < img_h:
            y_starts.append(max(0, img_h - tile_size))
        x_starts = list(range(0, max(1, img_w - tile_size + 1), stride))
        if not x_starts or x_starts[-1] + tile_size < img_w:
            x_starts.append(max(0, img_w - tile_size))

        tiled_boxes: List[Detection] = []
        for y1 in y_starts:
            y2 = min(img_h, y1 + tile_size)
            for x1 in x_starts:
                x2 = min(img_w, x1 + tile_size)
                tile = frame[y1:y2, x1:x2]
                if tile.shape[0] < 50 or tile.shape[1] < 50:
                    continue
                try:
                    res = coco_model(tile, conf=conf, imgsz=640, verbose=False)
                except Exception:
                    continue

                for b in res[0].boxes:
                    cid = int(b.cls[0])
                    cname = coco_model.names.get(cid, str(cid)).lower()
                    cval = float(b.conf[0])
                    canonical = _canonical_name(cname)
                    if canonical not in COCO_VEHICLE_NAMES:
                        continue

                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    fx1 = bx1 + x1
                    fy1 = by1 + y1
                    fx2 = bx2 + x1
                    fy2 = by2 + y1
                    fx1, fy1, fx2, fy2 = _clamp_box(fx1, fy1, fx2, fy2, img_w, img_h)

                    if _box_is_valid(fx1, fy1, fx2, fy2, img_w, img_h, canonical, self._debug):
                        tiled_boxes.append(Detection(
                            bbox=(fx1, fy1, fx2, fy2),
                            confidence=cval,
                            class_id=cid,
                            class_name=canonical,
                        ))

        return tiled_boxes

    # ── Inference ──────────────────────────────────────────────────────────────

    def detect_frame_with_persons(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        Run ensemble inference.
        Returns (vehicle_detections, person_detections).
        """
        if not self._models:
            return [], []

        if conf is None:
            conf = float(os.environ.get(
                "CONFIDENCE_THRESHOLD",
                self._det_cfg.get("confidence_threshold", 0.30)
            ))

        img_h, img_w = frame.shape[:2]
        imgsz = 1280 if max(img_h, img_w) >= 800 else 640

        all_vehicles: List[Detection] = []
        all_persons: List[Detection] = []

        t0 = time.perf_counter()
        with self._inference_lock:
            # ── 1. COCO Authority Model (yolov8m.pt) ───────────────────────────
            if "coco" in self._models:
                coco_model = self._models["coco"]
                try:
                    results = coco_model(frame, conf=max(conf, 0.20), iou=0.45, imgsz=imgsz, verbose=False)
                    for res in results:
                        if res.boxes is None:
                            continue
                        for box in res.boxes:
                            cid = int(box.cls[0])
                            raw_name = coco_model.names.get(cid, str(cid)).lower()
                            cval = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, img_w, img_h)

                            if raw_name == COCO_PERSON_NAME:
                                if _box_is_valid(x1, y1, x2, y2, img_w, img_h, "person", self._debug):
                                    all_persons.append(Detection(
                                        bbox=(x1, y1, x2, y2),
                                        confidence=cval,
                                        class_id=cid,
                                        class_name="person",
                                    ))
                            elif raw_name in COCO_VEHICLE_NAMES:
                                canonical = _canonical_name(raw_name)
                                if _box_is_valid(x1, y1, x2, y2, img_w, img_h, canonical, self._debug):
                                    all_vehicles.append(Detection(
                                        bbox=(x1, y1, x2, y2),
                                        confidence=cval,
                                        class_id=cid,
                                        class_name=canonical,
                                    ))
                except Exception as e:
                    print(f"[Detector] COCO inference error: {e}")

            # ── 2. SAHI-style Sliced Inference (TASK 2) ────────────────────────
            # Enabled if image is wider than 1280px OR full pass finds >= 8 bikes
            full_pass_bikes = [d for d in all_vehicles if d.class_name in ("motorcycle", "bicycle")]
            if img_w > 1280 or len(full_pass_bikes) >= 8:
                try:
                    sliced_dets = self._sliced_inference(frame, conf=max(conf, 0.20))
                    all_vehicles.extend(sliced_dets)
                except Exception as e:
                    print(f"[Detector] Sliced inference error: {e}")

            # ── 3. Custom models: ADD two-wheeler recall at lower threshold ───────
            CUSTOM_CONF = max(conf * 0.75, 0.18)

            for model_name in ("aerial", "ground", "cycles"):
                if model_name not in self._models:
                    continue
                model = self._models[model_name]
                try:
                    results = model(frame, conf=CUSTOM_CONF, iou=0.45, imgsz=imgsz, verbose=False)
                    for res in results:
                        if res.boxes is None:
                            continue
                        for box in res.boxes:
                            cid = int(box.cls[0])
                            raw_name = model.names.get(cid, str(cid)).lower()
                            cval = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, img_w, img_h)

                            canonical = _canonical_name(raw_name)
                            if canonical not in ("motorcycle", "bicycle"):
                                continue

                            if _box_is_valid(x1, y1, x2, y2, img_w, img_h, canonical, self._debug):
                                all_vehicles.append(Detection(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=cval,
                                    class_id=cid,
                                    class_name=canonical,
                                ))
                except Exception as e:
                    print(f"[Detector] {model_name} inference error: {e}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._update_perf(elapsed_ms)

        # ── NMS deduplication with Containment Suppression ─────────────────────
        vehicles = _nms(all_vehicles, same_class_iou=0.55, cross_class_iou=0.40)
        persons = _nms(all_persons, same_class_iou=0.50, cross_class_iou=0.50)

        # Final confidence filter: vehicle detections must meet confidence threshold
        vehicles = [d for d in vehicles if d.confidence >= conf]
        vehicles.sort(key=lambda d: d.confidence, reverse=True)

        return vehicles, persons

    def detect_frame(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """Backward-compatible: returns only vehicle detections."""
        vehicles, _ = self.detect_frame_with_persons(frame, conf=conf)
        return vehicles

    def detect_image(self, image_path: str, conf: Optional[float] = None) -> List[Detection]:
        """Load image from disk and run detection."""
        if not CV2_AVAILABLE:
            raise RuntimeError("opencv-python is required: pip install opencv-python-headless")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.detect_frame(frame, conf=conf)

    def reload_model(self, custom_path: Optional[str] = None) -> None:
        """Hot-reload the models."""
        with self._inference_lock:
            self._models.clear()
            self._load_all_models()

    # ── Performance tracking ───────────────────────────────────────────────────

    def _update_perf(self, elapsed_ms: float) -> None:
        self._frame_count += 1
        self._total_inference_ms += elapsed_ms
        elapsed_wall = time.time() - self._fps_start
        if elapsed_wall >= 2.0:
            self._last_fps = self._frame_count / elapsed_wall
            self._frame_count = 0
            self._total_inference_ms = 0.0
            self._fps_start = time.time()

    def get_stats(self) -> Dict[str, object]:
        avg_ms = (
            self._total_inference_ms / max(self._frame_count, 1)
            if self._frame_count else 0.0
        )
        return {
            "fps": round(self._last_fps, 1),
            "avg_inference_ms": round(avg_ms, 1),
            "models_loaded": list(self._models.keys()),
            "model_loaded": len(self._models) > 0,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
        }


# ── CLI test mode ──────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Two-wheeler photo detector")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if args.debug:
        os.environ["DEBUG_DETECT"] = "1"

    detector = TwoWheelerDetector.get_instance()
    print(f"\nModels loaded: {list(detector._models.keys())}")
    print(f"Running detection on: {args.image}")
    print(f"Confidence threshold: {args.conf:.0%}")

    detections, persons = detector.detect_frame_with_persons(
        cv2.imread(args.image), conf=args.conf
    )
    print(f"\nVehicles: {len(detections)}, Persons: {len(persons)}\n")

    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det.bbox
        print(f"  [{i}] {det.class_name:12s} | Conf: {det.confidence:.1%} | BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")


if __name__ == "__main__":
    _cli()
