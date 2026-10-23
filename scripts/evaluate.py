#!/usr/bin/env python3
"""
scripts/evaluate.py — Evaluate the parking detection pipeline on the test set.

Runs process_single_photo() on every image in tests/eval/images/,
checks for strip boxes, prints per-class detections, compares against visible counts,
and saves overlays to data/debug/.

Usage:
    python scripts/evaluate.py [--conf 0.30] [--debug]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def check_strip_box(bbox, img_w, img_h):
    """Return True if box is a malformed strip."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1; h = y2 - y1
    if w <= 0 or h <= 0:
        return True
    area_frac = (w * h) / max(img_w * img_h, 1)
    aspect = w / max(h, 1)
    if area_frac > 0.25:
        return True
    if aspect > 4.5 or aspect < 0.22:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate parking detector on test set")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--images-dir", default="tests/eval/images", help="Directory of test images")
    parser.add_argument("--manual-counts", default="tests/eval/manual_counts.json", help="Path to manual_counts.json")
    parser.add_argument("--output-dir", default="data/debug", help="Output overlays directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if args.debug:
        import os
        os.environ["DEBUG_DETECT"] = "1"

    from ml.detection.detect_photo import process_single_photo

    images_dir = PROJECT_ROOT / args.images_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manual counts
    counts_file = PROJECT_ROOT / args.manual_counts
    manual_counts = {}
    if counts_file.exists():
        try:
            with open(counts_file) as f:
                manual_counts = json.load(f)
        except Exception:
            manual_counts = {}

    image_files = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.jpeg")) +
        list(images_dir.glob("*.png"))
    )
    if not image_files:
        print(f"[eval] No images found in {images_dir}")
        sys.exit(1)

    print(f"\n[eval] Testing {len(image_files)} images (conf >= {args.conf:.0%})")
    print(f"[eval] Saving annotated debug overlays to: {output_dir}/")
    print("=" * 120)
    print(f"{'Image':<28} {'Classes Detected':<22} {'Total':<6} {'Visible':<10} {'Vio':<5} {'Move':<5} {'Unver':<6} {'Strip':<6} {'Time':<6}")
    print("-" * 120)

    results_table = []
    total_strip = 0
    total_vio = 0
    total_moving = 0
    total_unverified = 0
    total_detected = 0
    total_time = 0.0

    for img_path in image_files:
        img_name = img_path.name
        img_info = manual_counts.get(img_name, {})
        manual_val = img_info.get("manual_visible_count")
        auto_val = img_info.get("auto_estimated_count")

        t0 = time.time()
        try:
            out_path = str(output_dir / f"eval_{img_path.stem}_annotated.jpg")
            res = process_single_photo(
                image_path=str(img_path),
                conf_threshold=args.conf,
                output_path=out_path,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            detected = res["total_detected"]
            total_detected += detected
            vio = res["violations_count"]
            moving = res.get("moving_count", 0)
            unverified = res.get("unverified_count", 0)
            safe = res["safe_count"]

            # Count per class
            class_counts = {}
            for d in res["detections"]:
                c = d["class_name"]
                class_counts[c] = class_counts.get(c, 0) + 1

            # Format classes string
            c_parts = []
            for c in ["motorcycle", "bicycle", "car", "truck", "bus"]:
                if c in class_counts:
                    short_c = "moto" if c == "motorcycle" else ("bike" if c == "bicycle" else c)
                    c_parts.append(f"{short_c}:{class_counts[c]}")
            for c, cnt in class_counts.items():
                if c not in ["motorcycle", "bicycle", "car", "truck", "bus"]:
                    c_parts.append(f"{c}:{cnt}")
            class_str = " ".join(c_parts) if c_parts else "none"

            # Check strip boxes
            if CV2_AVAILABLE:
                frame = cv2.imread(str(img_path))
                img_h, img_w = frame.shape[:2] if frame is not None else (1080, 1920)
            else:
                img_w, img_h = 1920, 1080

            strip_count = sum(
                1 for item in res["detections"]
                if check_strip_box(item["bbox"], img_w, img_h)
            )

            total_strip += strip_count
            total_vio += vio
            total_moving += moving
            total_unverified += unverified

            # Visible count display
            if manual_val is not None:
                vis_str = f"{manual_val} (man)"
            elif auto_val is not None:
                vis_str = f"~{auto_val} (auto)"
            else:
                vis_str = f"~{detected} (auto)"
                # Record auto estimate
                if img_name not in manual_counts:
                    manual_counts[img_name] = {}
                manual_counts[img_name]["auto_estimated_count"] = detected
                manual_counts[img_name]["manual_visible_count"] = None

            strip_str = f"⚠️ {strip_count}" if strip_count > 0 else "0"
            print(
                f"{img_name:<28} "
                f"{class_str:<22} "
                f"{detected:<6} "
                f"{vis_str:<10} "
                f"{vio:<5} "
                f"{moving:<5} "
                f"{unverified:<6} "
                f"{strip_str:<6} "
                f"{elapsed:.2f}s"
            )

            results_table.append({
                "image": img_name,
                "detected": detected,
                "classes": class_counts,
                "visible_count": manual_val if manual_val is not None else auto_val,
                "violations": vio,
                "moving": moving,
                "unverified": unverified,
                "strip_boxes": strip_count,
                "elapsed_s": round(elapsed, 2),
            })

        except Exception as e:
            print(f"{img_name:<28} ERROR: {e}")
            results_table.append({"image": img_name, "error": str(e)})

    # Update manual_counts.json with auto estimates
    try:
        with open(counts_file, "w") as f:
            json.dump(manual_counts, f, indent=2)
    except Exception:
        pass

    # Summary
    print("-" * 120)
    avg_time = total_time / max(1, len(image_files))
    unverified_pct = (total_unverified / max(1, total_detected)) * 100

    print(f"\n[eval] SUMMARY ({len(image_files)} test images):")
    print(f"  • Total vehicles detected : {total_detected}")
    print(f"  • Total confirmed violations : {total_vio}")
    print(f"  • Total moving riders      : {total_moving} (suppressed, never red)")
    print(f"  • Total unverified         : {total_unverified} ({unverified_pct:.1f}% of detections) {'✅ <= 10%' if unverified_pct <= 10 else '⚠️ > 10%'}")
    print(f"  • Total STRIP BOXES        : {total_strip} {'✅ 0 STRIPS' if total_strip == 0 else '❌ HAS STRIPS'}")
    print(f"  • Average inference time   : {avg_time:.2f}s per image {'✅ < 10s' if avg_time < 10 else '❌ > 10s'}")
    print(f"  • Total pipeline run time  : {total_time:.1f}s")
    print("=" * 120)

    # Save JSON report
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "conf_threshold": args.conf,
            "num_images": len(image_files),
            "total_detected": total_detected,
            "total_strip_boxes": total_strip,
            "total_violations": total_vio,
            "total_moving": total_moving,
            "total_unverified": total_unverified,
            "unverified_percentage": round(unverified_pct, 1),
            "average_time_s": round(avg_time, 2),
            "per_image": results_table,
        }, f, indent=2)
    print(f"[eval] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
