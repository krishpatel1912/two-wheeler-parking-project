import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/{filename}")
@router.head("/{filename}")
def get_evidence_file(filename: str):
    project_root = Path(__file__).resolve().parents[4]
    evidence_dir = project_root / settings.evidence_dir
    file_path = evidence_dir / filename

    if not file_path.exists() or not file_path.is_file():
        # Check raw_images
        raw_path = project_root / "data" / "raw_images" / filename
        if raw_path.exists() and raw_path.is_file():
            file_path = raw_path
        else:
            # Fallback: maybe filename is full relative path
            alt_path = project_root / filename
            if alt_path.exists() and alt_path.is_file():
                file_path = alt_path
            else:
                raise HTTPException(status_code=404, detail="Evidence snapshot not found")

    ext = file_path.suffix.lower()
    media_type = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(str(file_path), media_type=media_type)
