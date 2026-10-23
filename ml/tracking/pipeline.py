"""
pipeline.py — End-to-End Video Processing Pipeline
===================================================
Reads video (file / webcam / single image), runs detection + tracking,
detects parking events, extracts features, and writes results to:
  - data/evidence/<timestamp>_track<id>.jpg  (annotated snapshots)
  - data/features/features.csv               (one row per parking event)

If a `db_callback` is provided (from the FastAPI backend), parking events
are also written to the database.

Usage:
    python ml/tracking/pipeline.py --source data/raw_video/myvideo.mp4
    python ml/tracking/pipeline.py --source 0 --show
    python ml/tracking/pipeline.py --source data/raw_images/frame.jpg
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[Pipeline] ERROR: opencv-python-headless not installed. Run: pip install opencv-python-headless")
    cv2 = None  # type: ignore

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ── Project root detection ─────────────────────────────────────────────────────

def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config" / "pipeline.yaml").exists():
            return parent
    return here.parents[2]


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_config(config_path: Optional[str] = None) -> dict:
    candidates = [
        Path(config_path) if config_path else None,
        PROJECT_ROOT / "config" / "pipeline.yaml",
        Path("config/pipeline.yaml"),
    ]
    for c in candidates:
        if c and c.exists() and YAML_AVAILABLE:
            with open(c) as f:
                return yaml.safe_load(f) or {}
    return {}


# ── Drawing helpers ────────────────────────────────────────────────────────────

COLORS = {
    "moving":    (0, 200, 0),    # green
    "parked":    (0, 0, 220),    # red
    "risk":      (0, 140, 255),  # orange
    "violation": (0, 0, 220),    # red
    "safe":      (0, 200, 0),    # green
}


def _draw_track(frame: np.ndarray, track, label: str = "") -> None:
    """Draw clear high-visibility bounding box + bilingual label on frame."""
    if not CV2_AVAILABLE:
        return
    x1, y1, x2, y2 = [int(v) for v in track.bbox]
    cls_name = getattr(track, "class_name", "").lower()

    # Determine status and color
    if cls_name == "vacant":
        color = (255, 191, 0)  # Cyan / Light Blue
        tag = f"#{track.track_id} [O] KHALI SLOT (VACANT)"
    elif label == "violation":
        color = (36, 36, 235)  # Bright Red
        tag = f"#{track.track_id} [X] GALAT PARKED (VIOLATION)"
    elif label == "risk":
        color = (0, 165, 255)  # Bright Amber / Orange
        tag = f"#{track.track_id} [!] RISK (MISALIGNED)"
    elif label == "safe":
        color = (46, 204, 113)  # Bright Emerald Green
        tag = f"#{track.track_id} [OK] SAHI PARKED (LEGAL)"
    else:
        color = (0, 200, 0)
        tag = f"#{track.track_id} {cls_name.upper()}"

    # Draw thick bounding box in designated status color
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Draw pill label background
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    top_y = max(44 + th, y1 + th + 4)
    left_x = max(4, min(x1, frame.shape[1] - tw - 14))
    cv2.rectangle(frame, (left_x, top_y - th - 6), (left_x + tw + 10, top_y + 4), color, -1)
    cv2.putText(frame, tag, (left_x + 5, top_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)



def _save_evidence(frame: np.ndarray, track, evidence_dir: Path) -> Optional[str]:
    """Save annotated snapshot for a parking event. Returns relative path."""
    if not CV2_AVAILABLE:
        return None
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_track{track.track_id}.jpg"
    fpath = evidence_dir / fname
    annotated = frame.copy()
    _draw_track(annotated, track)
    cv2.imwrite(str(fpath), annotated)
    return str(fpath.relative_to(PROJECT_ROOT)) if fpath.is_relative_to(PROJECT_ROOT) else str(fpath)


# ── VideoProcessor ─────────────────────────────────────────────────────────────

class VideoProcessor:
    """
    End-to-end video / image processor.

    Parameters
    ----------
    config : dict
        Loaded pipeline.yaml config.
    db_callback : callable, optional
        Called with (track, features, snapshot_path) for each new parking event.
        Used by the FastAPI backend to write to the database.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        db_callback: Optional[Callable] = None,
    ) -> None:
        self._config = config or _load_config()
        self._db_callback = db_callback
        self._paths_cfg = self._config.get("paths", {})
        self._output_cfg = self._config.get("output", {})

        self._evidence_dir = PROJECT_ROOT / self._paths_cfg.get("evidence_dir", "data/evidence")
        self._features_csv = PROJECT_ROOT / self._paths_cfg.get("features_csv", "data/features/features.csv")

    def process(
        self,
        source: str,
        output_video: Optional[str] = None,
        show: bool = False,
        camera_id: Optional[str] = None,
    ) -> Dict:
        """
        Process a video file, camera index, or single image.

        Returns a summary dict with counts.
        """
        # Import here to avoid circular issues
        from ml.detection.detector import TwoWheelerDetector
        from ml.tracking.tracker import ParkingTracker
        from ml.feature_engineering.features import FeatureExtractor
        from ml.classification.classifier import ParkingClassifier

        detector  = TwoWheelerDetector.get_instance()
        tracker   = ParkingTracker(self._config)
        extractor = FeatureExtractor(
            zone_config_path=str(PROJECT_ROOT / self._config.get("paths", {}).get("zone_config", "config/zones.json")),
            config=self._config,
        )
        classifier = ParkingClassifier.get_instance()

        # Detect if source is a single image
        source_path = Path(source) if not source.isdigit() else None
        is_image = source_path and source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if is_image:
            return self._process_image(source, detector, extractor, classifier)

        return self._process_video(source, output_video, show, detector, tracker, extractor, classifier)

    # ── Image processing ───────────────────────────────────────────────────────

    def _process_image(self, image_path: str, detector, extractor, classifier) -> Dict:
        if not CV2_AVAILABLE:
            raise RuntimeError("cv2 required for image processing")

        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Run detection with at least 60% accuracy threshold (configurable)
        conf_thresh = float(os.environ.get("CONFIDENCE_THRESHOLD", self._config.get("detection", {}).get("confidence_threshold", 0.60)))
        detections = detector.detect_frame(frame, conf=conf_thresh)
        print(f"[Pipeline] Image: {len(detections)} vehicle(s) detected (conf >= {conf_thresh:.0%})")

        from ml.tracking.tracker import TrackedVehicle

        # Convert to TrackedVehicles for feature engineering
        all_tracks = [
            TrackedVehicle(
                track_id=i + 1,
                bbox=det.bbox,
                class_name=det.class_name,
                confidence=det.confidence,
                is_parked=True,
                dwell_seconds=0.0,
            )
            for i, det in enumerate(detections)
        ]

        # First pass: extract features and classify each vehicle
        analyzed_events = []
        for i, tv in enumerate(all_tracks):
            other_tracks = [t for j, t in enumerate(all_tracks) if j != i]
            feat_row = extractor.extract(tv, frame, other_tracks)
            label_result = classifier.predict(feat_row.to_dict())
            feat_row.label = label_result.get("label", "pending")
            extractor.append_to_csv(feat_row, str(self._features_csv))
            analyzed_events.append((tv, feat_row, label_result))

        # Generate ONE master annotated photo showing ALL detections together
        master_snapshot_path = None
        if self._output_cfg.get("save_evidence_snapshots", True):
            annotated_frame = frame.copy()
            for tv, feat_row, label_res in analyzed_events:
                _draw_track(annotated_frame, tv, feat_row.label)

            # Draw header banner on master photo
            sahi_cnt = sum(1 for tv, feat, lres in analyzed_events if getattr(tv, "class_name", "").lower() != "vacant" and feat.label == "safe")
            galat_cnt = sum(1 for tv, feat, lres in analyzed_events if feat.label == "violation")
            khali_cnt = sum(1 for tv, feat, lres in analyzed_events if getattr(tv, "class_name", "").lower() == "vacant")
            risk_cnt = sum(1 for tv, feat, lres in analyzed_events if feat.label == "risk")

            banner = f"SAHI PARKED: {sahi_cnt}   |   GALAT PARKED (VIOLATION): {galat_cnt}   |   KHALI SLOTS: {khali_cnt}   |   RISK: {risk_cnt}"
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 36), (15, 23, 42), -1)
            cv2.putText(annotated_frame, banner, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

            self._evidence_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{ts}_photo_annotated.jpg"
            fpath = self._evidence_dir / fname
            cv2.imwrite(str(fpath), annotated_frame)
            master_snapshot_path = str(fpath.relative_to(PROJECT_ROOT)) if fpath.is_relative_to(PROJECT_ROOT) else str(fpath)

        # Notify DB callback with unified snapshot
        for tv, feat_row, label_res in analyzed_events:
            if self._db_callback:
                self._db_callback(tv, feat_row, master_snapshot_path, label_res)

        return {
            "source": image_path,
            "detections": len(detections),
            "parking_events": len(analyzed_events),
            "master_snapshot": master_snapshot_path,
            "confidence_threshold": conf_thresh,
        }

    # ── Video processing ───────────────────────────────────────────────────────

    def _process_video(
        self, source: str, output_video: Optional[str], show: bool,
        detector, tracker, extractor, classifier
    ) -> Dict:
        if not CV2_AVAILABLE:
            raise RuntimeError("cv2 required for video processing")

        # Open source
        cap_source = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[Pipeline] Source: {source}")
        print(f"[Pipeline] Resolution: {width}x{height} @ {fps:.1f} fps  ({total} frames)")

        # Optional output writer
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # State
        frames_processed   = 0
        total_detections   = 0
        parking_events     = 0
        parked_track_ids_seen: Set[int] = set()   # IDs already saved as evidence

        t_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # ── Detect + Track ─────────────────────────────────────────────
                detections = detector.detect_frame(frame)
                active_tracks = tracker.update(frame, detections)
                total_detections += len(detections)
                frames_processed += 1

                # ── Process newly-parked vehicles ──────────────────────────────
                parked_tracks = [t for t in active_tracks if t.is_parked]
                for track in parked_tracks:
                    if track.track_id not in parked_track_ids_seen:
                        parked_track_ids_seen.add(track.track_id)

                        # Feature extraction
                        other_parked = [t for t in parked_tracks if t.track_id != track.track_id]
                        feat_row = extractor.extract(track, frame, other_parked)

                        # Classification
                        label_result = classifier.predict(feat_row.to_dict())
                        feat_row.label = label_result.get("label", "pending")

                        # Save evidence snapshot
                        if self._output_cfg.get("save_evidence_snapshots", True):
                            snapshot_path = _save_evidence(frame, track, self._evidence_dir)
                        else:
                            snapshot_path = None

                        # Write feature row to CSV
                        extractor.append_to_csv(feat_row, str(self._features_csv))
                        parking_events += 1

                        # Callback for DB (used by FastAPI backend)
                        if self._db_callback:
                            self._db_callback(track, feat_row, snapshot_path, label_result)

                        print(
                            f"[Pipeline] Parking event: track={track.track_id}"
                            f"  dwell={track.dwell_seconds:.0f}s"
                            f"  label={feat_row.label}"
                            f"  zone={feat_row.zone_id}"
                        )

                # ── Draw overlays ──────────────────────────────────────────────
                if show or writer:
                    annotated = frame.copy()
                    for track in active_tracks:
                        label = ""
                        # Show classification label if this track has been processed
                        if track.track_id in parked_track_ids_seen:
                            # Re-classify from saved CSV would be expensive; just show "parked"
                            label = "parked"
                        _draw_track(annotated, track, label)

                    # Stats overlay
                    cv2.putText(annotated,
                                f"Frame {frames_processed}  Detected: {len(detections)}  Parked events: {parking_events}",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if writer:
                        writer.write(annotated)
                    if show:
                        cv2.imshow("Two-Wheeler Parking Detection", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("[Pipeline] Quit requested")
                            break

                # Progress report every 100 frames
                if frames_processed % 100 == 0:
                    elapsed = time.time() - t_start
                    speed = frames_processed / elapsed if elapsed > 0 else 0
                    pct = (frames_processed / total * 100) if total > 0 else 0
                    print(f"[Pipeline] {frames_processed}/{total} frames ({pct:.0f}%)  {speed:.1f} fps")

        except KeyboardInterrupt:
            print("[Pipeline] Interrupted")
        finally:
            cap.release()
            if writer:
                writer.release()
            if show and CV2_AVAILABLE:
                cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        summary = {
            "source": source,
            "frames_processed": frames_processed,
            "total_detections": total_detections,
            "parking_events": parking_events,
            "elapsed_seconds": round(elapsed, 1),
            "avg_fps": round(frames_processed / elapsed, 1) if elapsed > 0 else 0,
            "features_csv": str(self._features_csv),
        }
        print("\n[Pipeline] Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run two-wheeler parking detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video file path, image file path, or camera index (e.g. 0)",
    )
    parser.add_argument("--output-video", default=None, help="Save annotated video to this path")
    parser.add_argument("--show", action="store_true", help="Display video window")
    parser.add_argument("--config", default=None, help="Path to pipeline.yaml")
    parser.add_argument("--device", default=None, help="Override inference device (cpu/cuda)")
    args = parser.parse_args()

    if args.device:
        os.environ["DEVICE"] = args.device

    config = _load_config(args.config)
    processor = VideoProcessor(config=config)
    processor.process(
        source=args.source,
        output_video=args.output_video,
        show=args.show,
    )


if __name__ == "__main__":
    main()
