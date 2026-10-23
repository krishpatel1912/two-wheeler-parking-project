"""
detect_photo.py — Single Photo Two-Wheeler Detection & Parking Analysis (v4)
=============================================================================
Conservative 3-stage pipeline:
  Stage 0: Rider suppression (person overlapping bike → MOVING, grey, no violation)
  Stage 1: Alien vehicle check (car/truck/bus in bike area → red VIOLATION)
  Stage 2: Row analysis — ONLY if >= 5 two-wheelers AND >= 2 valid row clusters.
           Otherwise → UNVERIFIED (amber), never a false violation.
  Stage 3: Everything else → SAFE (green)

Confidence gates:
  - Alien vehicle violation: conf >= 0.40
  - Two-wheeler row violation: conf >= 0.35
  - Below thresholds → UNVERIFIED (amber)

Box labels on image: SHORT (≤20 chars).
Full reason text in API response only.

Usage:
    python ml/detection/detect_photo.py --image data/raw_images/my_photo.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("ERROR: opencv-python is required. Install: pip install opencv-python-headless")

from ml.detection.detector import TwoWheelerDetector, Detection
from ml.zone_config.zone_loader import ZoneConfig, ZoneLoader
from ml.feature_engineering.geometry import (
    point_in_polygon,
    polygon_distance,
    bbox_orientation_angle,
    bbox_overlaps_polygon,
    centroid_of_bbox,
)


# ── Color palette (BGR for OpenCV) ─────────────────────────────────────────────
CV_COLORS = {
    "violation":  (36,  36,  235),   # Bright Red
    "risk":       (0,   165, 255),   # Orange
    "moving":     (128, 128, 128),   # Grey
    "unverified": (0,   200, 220),   # Amber/cyan
    "safe":       (46,  204, 113),   # Green (not drawn)
    "neutral":    (235, 140, 0),
}

# Short labels shown ON the image (≤20 chars)
SHORT_LABELS = {
    "Four-Wheeler in Bay":            "Car in bike bay",
    "Bicycle in Bay":                 "Bicycle here",
    "Standing in Walking Path":       "Blocking path",
    "Non-Sequence Row":               "Out of row",
    "Parked Sideways":                "Parked sideways",
    "Active Rider Detected":          "Moving (rider)",
    "Properly Parked in Row":         "Legal parking",
    "Unverified":                     "Unverified",
}


def _box_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = aA + aB - inter
    return inter / union if union > 0 else 0.0


def _person_overlaps_vehicle(person: Detection, vehicle: Detection) -> bool:
    """True if person center is inside the vehicle box OR IoU > 0.20."""
    px, py = person.center
    vx1, vy1, vx2, vy2 = vehicle.bbox
    # Center inside box
    if vx1 <= px <= vx2 and vy1 <= py <= vy2:
        return True
    # Significant overlap
    if _box_iou(person.bbox, vehicle.bbox) > 0.20:
        return True
    return False


def analyze_all_photo_detections(
    detections: List[Detection],
    zones: List[ZoneConfig],
    frame_shape: Tuple[int, int],
    px_per_meter: float = 50.0,
    persons: Optional[List[Detection]] = None,
) -> List[Dict[str, Any]]:
    """
    Conservative 3-stage parking analysis.

    Returns one result dict per detection (same order as input detections).
    Result dict keys:
      status:   "violation" | "moving" | "unverified" | "safe" | "risk"
      reason:   full human-readable text
      short:    short label for on-image drawing (≤20 chars)
      zone:     zone name
      inside_bay: bool
      is_moving:  bool (True for rider-suppressed vehicles)
    """
    if not detections:
        return []

    if persons is None:
        persons = []

    img_h, img_w = frame_shape[0], frame_shape[1]
    results_map: Dict[int, Dict[str, Any]] = {}

    # ── STAGE 0: Rider suppression ─────────────────────────────────────────────
    # For each two-wheeler, check if any person overlaps it.
    moving_ids: set = set()
    for det in detections:
        vid = id(det)
        cname = det.class_name.lower()
        if cname not in ("motorcycle", "bicycle", "scooter", "bike"):
            continue
        for person in persons:
            if _person_overlaps_vehicle(person, det):
                moving_ids.add(vid)
                results_map[vid] = {
                    "status": "moving",
                    "reason": "Active Rider Detected — Vehicle in Motion",
                    "short": "Moving (rider)",
                    "zone": "Road",
                    "inside_bay": False,
                    "is_moving": True,
                }
                break

    # ── STAGE 1: Alien vehicle check ───────────────────────────────────────────
    # Only runs for car/truck/bus/bicycle (not two-wheelers or moving vehicles)
    for det in detections:
        vid = id(det)
        if vid in results_map:
            continue  # already classified (moving)

        cname = det.class_name.lower()

        if cname in ("car", "truck", "bus"):
            if det.confidence >= 0.40:
                results_map[vid] = {
                    "status": "violation",
                    "reason": "Four-Wheeler / Car Parked in Two-Wheeler Bay",
                    "short": "Car in bike bay",
                    "zone": "Two-Wheeler Bay",
                    "inside_bay": False,
                    "is_moving": False,
                }
            else:
                results_map[vid] = {
                    "status": "unverified",
                    "reason": f"Possible car detected (conf {det.confidence:.0%} < 40% threshold)",
                    "short": "Unverified vehicle",
                    "zone": "Open Area",
                    "inside_bay": False,
                    "is_moving": False,
                }
            continue

        if cname == "bicycle":
            if det.confidence >= 0.40:
                results_map[vid] = {
                    "status": "violation",
                    "reason": "Bicycle Parked in Motorized-Vehicle Bay",
                    "short": "Bicycle here",
                    "zone": "Vehicle Bay",
                    "inside_bay": False,
                    "is_moving": False,
                }
            else:
                results_map[vid] = {
                    "status": "unverified",
                    "reason": f"Possible bicycle (conf {det.confidence:.0%} < 40% threshold)",
                    "short": "Unverified",
                    "zone": "Open Area",
                    "inside_bay": False,
                    "is_moving": False,
                }
            continue

    # Collect unclassified two-wheelers
    two_wheelers = [
        d for d in detections
        if id(d) not in results_map and d.class_name.lower() in ("motorcycle", "scooter", "bike")
    ]

    if not two_wheelers:
        return [results_map.get(id(d), {
            "status": "safe", "reason": "Properly Parked", "short": "Legal parking",
            "zone": "Parking Row", "inside_bay": True, "is_moving": False,
        }) for d in detections]

    # ── STAGE 2: Row analysis (CONSERVATIVE) ──────────────────────────────────
    # ONLY runs if >= 5 two-wheelers AND can form >= 2 row clusters of >= 2 each.
    # Otherwise mark all two-wheelers UNVERIFIED (amber) — never a false violation.

    def angle_delta(a1: float, a2: float) -> float:
        d = abs(a1 - a2) % 180.0
        return 180.0 - d if d > 90.0 else d

    med_h = float(np.median([d.height for d in two_wheelers]))
    row_tolerance = max(med_h * 0.60, 35.0)

    # Cluster into rows by Y-center proximity
    rows: List[List[Detection]] = []
    for veh in sorted(two_wheelers, key=lambda d: d.center[1]):
        placed = False
        for row in rows:
            row_y = float(np.mean([v.center[1] for v in row]))
            if abs(veh.center[1] - row_y) <= row_tolerance:
                row.append(veh)
                placed = True
                break
        if not placed:
            rows.append([veh])

    # Valid rows: clusters with >= 3 vehicles forming a true parking row
    valid_rows = [r for r in rows if len(r) >= 3]
    n_tw = len(two_wheelers)

    # Conservative gate: need >= 5 two-wheelers AND >= 1 well-formed row with >= 3 bikes
    row_analysis_ok = (n_tw >= 5) and (len(valid_rows) >= 1)

    if not row_analysis_ok:
        # Not enough row context — bikes with conf >= 0.50 are SAFE (no red box);
        # only marginal detections (0.30 - 0.50) are marked UNVERIFIED
        for det in two_wheelers:
            if det.confidence >= 0.50:
                results_map[id(det)] = {
                    "status": "safe",
                    "reason": "Properly Parked",
                    "short": "Legal parking",
                    "zone": "Open Lot",
                    "inside_bay": True,
                    "is_moving": False,
                }
            else:
                results_map[id(det)] = {
                    "status": "unverified",
                    "reason": f"Unverified positioning ({n_tw} bikes in lot, conf {det.confidence:.0%})",
                    "short": "Unverified",
                    "zone": "Open Lot",
                    "inside_bay": False,
                    "is_moving": False,
                }
    else:
        # Full row analysis with conservative rules
        # Build row_bands for membership lookup
        row_bands = []
        for row in rows:
            row_angles = [bbox_orientation_angle(v.bbox) for v in row]
            row_bands.append({
                "center_y": float(np.mean([v.center[1] for v in row])),
                "min_y": min(v.bbox[1] for v in row),
                "max_y": max(v.bbox[3] for v in row),
                "vehicles": row,
                "ids": {id(v) for v in row},
                "count": len(row),
                "med_angle": float(np.median(row_angles)) if row_angles else 90.0,
            })
        row_bands.sort(key=lambda rb: rb["center_y"])

        # Aisles: clear gaps between consecutive valid rows
        aisles = []
        for i in range(len(valid_rows) - 1):
            # Find the two adjacent valid rows
            gap_top  = max(v.bbox[3] for v in valid_rows[i])
            gap_bot  = min(v.bbox[1] for v in valid_rows[i + 1])
            gap_size = gap_bot - gap_top
            if gap_size > med_h * 0.45:
                aisles.append({"top": gap_top, "bottom": gap_bot})

        for veh in two_wheelers:
            vid = id(veh)
            if vid in results_map:
                continue

            cy = veh.center[1]
            v_ang = bbox_orientation_angle(veh.bbox)

            # Aisle obstruction (needs conf >= 0.35)
            in_aisle = any((a["top"] + 10) <= cy <= (a["bottom"] - 10) for a in aisles)
            if in_aisle:
                if veh.confidence >= 0.35:
                    results_map[vid] = {
                        "status": "violation",
                        "reason": "Standing in Walking Path / Aisle Passage",
                        "short": "Blocking path",
                        "zone": "Walking Path / Aisle",
                        "inside_bay": False,
                        "is_moving": False,
                    }
                else:
                    results_map[vid] = {
                        "status": "unverified",
                        "reason": f"Possible aisle obstruction (conf {veh.confidence:.0%} < 35%)",
                        "short": "Unverified",
                        "zone": "Aisle",
                        "inside_bay": False,
                        "is_moving": False,
                    }
                continue

            # Row membership check
            in_band = next((rb for rb in row_bands if vid in rb["ids"]), None)

            # Isolated vehicle NOT on an edge → Non-sequence check
            if in_band and in_band["count"] == 1 and len(valid_rows) >= 2 and n_tw >= 6:
                near_edge = (
                    veh.center[0] < img_w * 0.10
                    or veh.center[0] > img_w * 0.90
                    or cy < img_h * 0.10
                    or cy > img_h * 0.90
                )
                if not near_edge:
                    # Only flag non-sequence if bike is also noticeably misaligned (Δ > 25°)
                    lot_angle = in_band["med_angle"] if in_band else 90.0
                    ang_diff = angle_delta(v_ang, lot_angle)
                    if ang_diff > 25.0 and veh.confidence >= 0.40:
                        results_map[vid] = {
                            "status": "violation",
                            "reason": f"Vehicle Out of Parking Sequence & Misaligned (Δ{ang_diff:.0f}°)",
                            "short": "Out of row",
                            "zone": "Path / Passage",
                            "inside_bay": False,
                            "is_moving": False,
                        }
                        continue

            # Angle check (sideways parking)
            ref_angle = in_band["med_angle"] if in_band else 90.0
            a_dev = angle_delta(v_ang, ref_angle)

            if a_dev > 32.0 and veh.confidence >= 0.35:
                results_map[vid] = {
                    "status": "violation",
                    "reason": f"Parked Sideways / Wrong Angle (Δ{a_dev:.0f}° from row median)",
                    "short": "Parked sideways",
                    "zone": "Parking Row",
                    "inside_bay": False,
                    "is_moving": False,
                }
            elif a_dev > 22.0:
                results_map[vid] = {
                    "status": "risk",
                    "reason": f"Slightly Misaligned (Δ{a_dev:.0f}° from row median)",
                    "short": "Slight misalign",
                    "zone": "Parking Row",
                    "inside_bay": True,
                    "is_moving": False,
                }
            else:
                results_map[vid] = {
                    "status": "safe",
                    "reason": "Properly Parked in Row",
                    "short": "Legal parking",
                    "zone": "Parking Row",
                    "inside_bay": True,
                    "is_moving": False,
                }

    # ── Default: anything still unclassified is SAFE ──────────────────────────
    return [
        results_map.get(id(d), {
            "status": "safe",
            "reason": "Properly Parked",
            "short": "Legal parking",
            "zone": "Parking Row",
            "inside_bay": True,
            "is_moving": False,
        })
        for d in detections
    ]


def analyze_parking_status(
    det: Detection,
    zones: List[ZoneConfig],
    px_per_meter: float = 50.0,
) -> Dict[str, Any]:
    """Single vehicle backward-compatible wrapper."""
    return analyze_all_photo_detections([det], zones, (1080, 1920), px_per_meter)[0]


def draw_photo_detections(
    frame: np.ndarray,
    detections: List[Detection],
    analysis_results: List[Dict[str, Any]],
    zones: List[ZoneConfig],
) -> np.ndarray:
    """
    Draw annotated results on the frame.
    Colors:
      RED   = violation
      ORANGE = risk
      GREY  = moving (active rider)
      CYAN  = unverified (amber-ish)
      No box = safe (clean image)

    Labels: SHORT (≤20 chars) drawn on image.
    Full text available in API JSON only.
    """
    canvas = frame.copy()
    img_h, img_w = canvas.shape[:2]

    # 1. Draw Zone Polygons (only when zones are active)
    for z in zones:
        if len(z.boundary_polygon) >= 3:
            pts = np.array(z.boundary_polygon, dtype=np.int32)
            color = CV_COLORS.get("neutral", (200, 150, 0))
            if z.zone_type == "fire_lane":
                color = CV_COLORS["violation"]
            elif z.zone_type == "walkway":
                color = CV_COLORS["risk"]

            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
            cv2.polylines(canvas, [pts], True, color, 2)

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(canvas, z.name, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 2. Draw boxes for violations / moving / unverified / risk
    for idx, (det, analysis) in enumerate(zip(detections, analysis_results), 1):
        status = analysis.get("status", "safe")
        if status == "safe":
            continue  # leave legally parked vehicles clean

        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        # Safety clamp
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        # Pick color
        if status == "violation":
            color = (36, 36, 235)      # Red
            thickness = 3
        elif status == "moving":
            color = (120, 120, 120)    # Grey
            thickness = 2
        elif status == "unverified":
            color = (0, 200, 220)      # Cyan/amber
            thickness = 2
        else:  # risk
            color = (0, 165, 255)      # Orange
            thickness = 2

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Short label on box — clamp to image bounds, no truncation
        short_label = analysis.get("short", status)
        conf_pct = f"{det.confidence:.0%}"
        label_text = f"#{idx} {short_label} ({conf_pct})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        font_thick = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, font_thick)

        # Position: try above box, fall back to inside-top
        label_y = y1 - 6
        if label_y - th - 4 < 0:
            label_y = y1 + th + 6  # inside box top

        label_x = max(2, min(x1, img_w - tw - 8))

        # Background pill
        cv2.rectangle(canvas,
                      (label_x - 2, label_y - th - 4),
                      (label_x + tw + 6, label_y + 2),
                      color, -1)
        cv2.putText(canvas, label_text, (label_x + 2, label_y - 1),
                    font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    # 3. Header Banner
    vio_cnt       = sum(1 for a in analysis_results if a.get("status") == "violation")
    legal_cnt     = sum(1 for a in analysis_results if a.get("status") == "safe")
    moving_cnt    = sum(1 for a in analysis_results if a.get("status") == "moving")
    unverified_cnt = sum(1 for a in analysis_results if a.get("status") == "unverified")
    risk_cnt      = sum(1 for a in analysis_results if a.get("status") == "risk")

    banner = (f"Legal: {legal_cnt}  |  Violations: {vio_cnt}"
              f"  |  Moving: {moving_cnt}  |  Unverified: {unverified_cnt}"
              f"  |  Risk: {risk_cnt}")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (15, 23, 42), -1)
    cv2.putText(canvas, banner, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def process_single_photo(
    image_path: str,
    conf_threshold: float = 0.30,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    zones_path: Optional[str] = None,
    show_window: bool = False,
) -> Dict[str, Any]:
    """
    Process a single photo end-to-end.
    Returns full summary dict including per-detection items.
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required: pip install opencv-python-headless")

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(str(img_path))
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Load detector (singleton)
    detector = TwoWheelerDetector.get_instance()
    if model_path:
        detector.reload_model(model_path)

    # Load zones (only when an explicit zones path is provided)
    zones: List[ZoneConfig] = []
    if zones_path:
        zones = ZoneLoader.load(zones_path)

    # Run detection (with person detection for rider suppression)
    detections, persons = detector.detect_frame_with_persons(frame, conf=conf_threshold)

    # Evaluate parking status with rider suppression
    analysis_results = analyze_all_photo_detections(
        detections, zones, frame.shape[:2], persons=persons
    )

    # Annotate the photo
    annotated_frame = draw_photo_detections(frame, detections, analysis_results, zones)

    # Save output
    out_dir = PROJECT_ROOT / "data" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_path:
        out_file = Path(output_path)
    else:
        out_file = out_dir / f"{img_path.stem}_annotated.jpg"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), annotated_frame)

    # Build summary
    safe_cnt       = sum(1 for a in analysis_results if a.get("status") == "safe")
    risk_cnt       = sum(1 for a in analysis_results if a.get("status") == "risk")
    vio_cnt        = sum(1 for a in analysis_results if a.get("status") == "violation")
    moving_cnt     = sum(1 for a in analysis_results if a.get("status") == "moving")
    unverified_cnt = sum(1 for a in analysis_results if a.get("status") == "unverified")
    vacant_cnt     = sum(1 for a in analysis_results if a.get("status") == "vacant_slot")

    items = []
    img_h, img_w = frame.shape[:2]
    for idx, (det, analysis) in enumerate(zip(detections, analysis_results), 1):
        status = analysis.get("status", "safe")
        crop_url = None
        crop_path = None

        # For violations, risks, and unverified: save individual cropped thumbnail
        if status in ("violation", "risk", "unverified", "moving"):
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(img_w, x2 + pad_x)
            crop_y2 = min(img_h, y2 + pad_y)

            crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop_img.size > 0:
                crop_fname = f"{img_path.stem}_det_{idx}.jpg"
                crop_dest = out_dir / crop_fname
                cv2.imwrite(str(crop_dest), crop_img)
                crop_url = f"/api/v1/evidence/{crop_fname}"
                crop_path = str(crop_dest)

        items.append({
            "index":              idx,
            "class_name":         det.class_name,
            "confidence":         round(float(det.confidence), 4),
            "confidence_percent": f"{det.confidence:.1%}",
            "bbox":               [round(float(v), 1) for v in det.bbox],
            "status":             status,
            "reason":             analysis.get("reason", ""),
            "short":              analysis.get("short", status),
            "zone":               analysis.get("zone", "Open Lot"),
            "center":             [round(float(v), 1) for v in det.center],
            "is_moving":          analysis.get("is_moving", False),
            "crop_url":           crop_url,
            "crop_path":          crop_path,
        })

    summary = {
        "image_path":          str(img_path),
        "annotated_image":     str(out_file),
        "total_detected":      len(detections),
        "confidence_threshold": conf_threshold,
        "safe_count":          safe_cnt,
        "violations_count":    vio_cnt,
        "vacant_count":        vacant_cnt,
        "risk_count":          risk_cnt,
        "moving_count":        moving_cnt,
        "unverified_count":    unverified_cnt,
        "counts": {
            "safe":        safe_cnt,
            "risk":        risk_cnt,
            "violation":   vio_cnt,
            "vacant":      vacant_cnt,
            "moving":      moving_cnt,
            "unverified":  unverified_cnt,
        },
        "detections": items,
        "items":      items,
        "model_used": detector._model_path,
        "persons_detected": len(persons),
    }

    if show_window:
        cv2.imshow("Single Photo Parking Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return summary


# ── Terminal Output Formatter ──────────────────────────────────────────────────

def print_summary_table(res: Dict[str, Any]) -> None:
    total  = res["total_detected"]
    counts = res["counts"]
    print("\n" + "=" * 90)
    print(f"PHOTO DETECTION RESULTS — {Path(res['image_path']).name}")
    print("=" * 90)
    print(f"  Model:        {res['model_used']}")
    print(f"  Conf cut:     >= {res['confidence_threshold']:.0%}")
    print(f"  Detected:     {total} vehicles, {res.get('persons_detected', 0)} persons")
    print(f"  Summary:  🟢 Safe: {counts['safe']}  |  🔴 Violations: {counts['violation']}"
          f"  |  ⚪ Moving: {counts.get('moving', 0)}"
          f"  |  🟡 Unverified: {counts.get('unverified', 0)}"
          f"  |  🟠 Risk: {counts['risk']}")
    print("-" * 90)

    if total == 0:
        print("  No vehicles detected at this confidence threshold.")
        print("=" * 90 + "\n")
        return

    print(f"{'#':<3} {'Class':<12} {'Conf':<8} {'Status':<14} {'Zone':<20} {'Reason'}")
    print("-" * 90)

    STATUS_ICONS = {
        "safe": "🟢 SAFE",
        "risk": "🟠 RISK",
        "violation": "🔴 VIOLATION",
        "moving": "⚪ MOVING",
        "unverified": "🟡 UNVERIFIED",
        "vacant_slot": "🔵 VACANT",
    }
    for item in res["detections"]:
        tag = STATUS_ICONS.get(item["status"], item["status"].upper())
        print(
            f"{item['index']:<3} "
            f"{item['class_name']:<12} "
            f"{item['confidence_percent']:<8} "
            f"{tag:<14} "
            f"{item['zone']:<20} "
            f"{item['reason']}"
        )

    print("-" * 90)
    print(f"✓ Annotated photo saved to: {res['annotated_image']}")
    print("=" * 90 + "\n")


# ── CLI Interface ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservative Single-Photo Parking Violation Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", default="",
                        help="Path to single input photo (.jpg, .png)")
    parser.add_argument("--input-dir", "-d", default="",
                        help="Path to folder of photos to process")
    parser.add_argument("--conf", "-c", type=float, default=0.30,
                        help="Confidence threshold")
    parser.add_argument("--model", "-m", default="",
                        help="Path to model weights")
    parser.add_argument("--output", "-o", default="",
                        help="Path to save annotated photo")
    parser.add_argument("--zones", "-z", default="",
                        help="Path to config/zones.json")
    parser.add_argument("--show", action="store_true",
                        help="Display photo in desktop window")
    parser.add_argument("--json", action="store_true",
                        help="Print machine-readable JSON output")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug box filter output")
    args = parser.parse_args()

    if args.debug:
        os.environ["DEBUG_DETECT"] = "1"

    target_files = []
    if args.image:
        target_files.append(Path(args.image))
    elif args.input_dir:
        folder = Path(args.input_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            target_files.extend(list(folder.glob(ext)))
    else:
        raw_imgs = (
            list((PROJECT_ROOT / "data" / "raw_images").glob("*.jpg")) +
            list((PROJECT_ROOT / "data" / "raw_images").glob("*.png"))
        )
        if raw_imgs:
            print(f"No image specified. Processing sample image: {raw_imgs[0]}")
            target_files.append(raw_imgs[0])
        else:
            print("ERROR: Please specify an image with --image <path/to/photo.jpg>")
            sys.exit(1)

    all_results = []
    for f in target_files:
        try:
            res = process_single_photo(
                image_path=str(f),
                conf_threshold=args.conf,
                model_path=args.model,
                output_path=args.output if len(target_files) == 1 else "",
                zones_path=args.zones,
                show_window=args.show,
            )
            all_results.append(res)
            if not args.json:
                print_summary_table(res)
        except Exception as exc:
            print(f"ERROR processing {f}: {exc}")

    if args.json:
        print(json.dumps(all_results if len(all_results) > 1 else all_results[0], indent=2))


if __name__ == "__main__":
    main()
