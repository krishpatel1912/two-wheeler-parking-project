import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from app.api.deps import DbSession
from app.core.config import get_settings
from app.schemas.processing import ProcessingJobResponse
from app.services.pipeline_service import enqueue_processing_job, get_job, list_jobs

router = APIRouter()
settings = get_settings()


@router.post("/upload", response_model=ProcessingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_footage(
    file: UploadFile = File(...),
    camera_id: Optional[str] = Form(None),
) -> ProcessingJobResponse:
    ext = Path(file.filename).suffix.lower()
    is_video = ext in {".mp4", ".avi", ".mov", ".mkv"}
    is_image = ext in {".jpg", ".jpeg", ".png", ".webp"}

    if not (is_video or is_image):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Upload video (.mp4, .avi) or image (.jpg, .png).",
        )

    project_root = Path(__file__).resolve().parents[4]
    dest_dir = project_root / ("data/raw_video" if is_video else "data/raw_images")
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_filename = f"{uuid.uuid4()}{ext}"
    saved_path = dest_dir / saved_filename

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_res = enqueue_processing_job(
        file_path=str(saved_path),
        original_filename=file.filename,
        camera_id=camera_id,
    )
    return job_res


@router.post("/detect-photo")
async def detect_photo_endpoint(
    db: DbSession,
    file: UploadFile = File(...),
    conf: float = Form(0.30),
    camera_id: Optional[str] = Form(None),
):
    """
    Synchronous single-photo detection endpoint.
    Takes ANY random bike parking image, detects all vehicles,
    marks unpositioned / violated bikes in RED boxes, risk in AMBER,
    and returns ONE unified annotated image URL and metrics instantly.
    Also automatically adds violated vehicles to the Violations database section.
    """
    ext = Path(file.filename).suffix.lower()
    is_image = ext in {".jpg", ".jpeg", ".png", ".webp"}
    if not is_image:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Upload an image (.jpg, .jpeg, .png).",
        )

    project_root = Path(__file__).resolve().parents[4]
    raw_dir = project_root / "data/raw_images"
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved_filename = f"{uuid.uuid4()}{ext}"
    saved_path = raw_dir / saved_filename

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from ml.detection.detect_photo import process_single_photo
    result = process_single_photo(
        image_path=str(saved_path),
        conf_threshold=conf,
    )

    annotated_path = Path(result["annotated_image"])
    annotated_fname = annotated_path.name

    # ── Automatically persist detected violations and risk items to Violations DB ──
    try:
        from app.models.parking_event import ParkingEvent
        from app.models.vehicle import Vehicle
        from app.models.violation import Violation

        for item in result.get("detections", []):
            st = item.get("status")
            if st in ("violation", "risk"):
                # Use cropped thumbnail if available, otherwise annotated image
                v_thumb = Path(item["crop_path"]).name if item.get("crop_path") else annotated_fname
                
                # Create a vehicle record
                veh = Vehicle(
                    track_id=int(item.get("index", 1)),
                    class_name=item.get("class_name", "motorcycle"),
                    thumbnail_path=v_thumb,
                    camera_id=camera_id,
                )
                db.add(veh)
                db.flush()

                # Create ParkingEvent
                event = ParkingEvent(
                    vehicle_id=veh.id,
                    dwell_seconds=60.0,
                    features={
                        "bbox": item.get("bbox"),
                        "reason": item.get("reason"),
                        "zone": item.get("zone"),
                        "confidence": item.get("confidence"),
                        "crop_url": item.get("crop_url"),
                    },
                    ml_label=st,
                    ml_confidence=item.get("confidence"),
                    snapshot_path=v_thumb,
                )
                db.add(event)
                db.flush()

                # Create Violation entry for the Violations Section
                v_record = Violation(
                    parking_event_id=event.id,
                    label=st,
                    review_status="pending",
                    reviewer_notes=f"Single photo detection: {item.get('reason', 'Unpositioned parking')}",
                )
                db.add(v_record)

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[detect-photo] Could not persist violations to DB: {e}")

    return {
        "success": True,
        "filename": file.filename,
        "saved_filename": saved_filename,
        "original_image_url": f"/api/v1/evidence/{saved_filename}",
        "annotated_image_url": f"/api/v1/evidence/{annotated_fname}",
        "total_detected": result["total_detected"],
        "safe_count": result["safe_count"],
        "violations_count": result["violations_count"],
        "vacant_count": result["vacant_count"],
        "risk_count": result["risk_count"],
        "moving_count": result.get("moving_count", 0),
        "unverified_count": result.get("unverified_count", 0),
        "counts": result["counts"],
        "detections": result["detections"],
        "confidence_threshold": conf,
        "model_used": result.get("model_used", ""),
        "persons_detected": result.get("persons_detected", 0),
    }


@router.get("/jobs", response_model=List[ProcessingJobResponse])
def get_jobs_list() -> List[ProcessingJobResponse]:
    return list_jobs()


@router.get("/jobs/{job_id}", response_model=ProcessingJobResponse)
def get_job_status(job_id: str) -> ProcessingJobResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
