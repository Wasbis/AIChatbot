from fastapi import APIRouter, BackgroundTasks, Depends
import subprocess
import sys
import os
from sqlalchemy.orm import Session
from src.core.database import get_db, SessionLocal
from src.models.ingest_log import IngestLog

router = APIRouter()


def run_scraper_script(task_id: int):
    """Fungsi untuk menjalankan script scraper di background"""
    # Gunakan PYTHONPATH=. agar module src bisa ditemukan
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    try:
        subprocess.run(
            [sys.executable, "-u", "scripts/scrape_website.py", "--task-id", str(task_id)],
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
            check=True
        )
    except Exception as e:
        # Update status ke FAILED jika script gagal total
        db = SessionLocal()
        task = db.query(IngestLog).filter(IngestLog.id == task_id).first()
        if task:
            task.status = "FAILED"
            task.message = f"Critical Error: {str(e)}"
            db.commit()
        db.close()


@router.post("/website")
def ingest_website(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Endpoint untuk menyuruh AI membaca ulang website.
    Dijalankan di background agar API tidak timeout/nge-hang.
    """
    # 1. Buat log entry baru
    new_task = IngestLog(status="RUNNING", message="Inisialisasi crawler...")
    db.add(new_task)
    db.commit()
    db.refresh(new_task)

    # 2. Jalankan task di background
    background_tasks.add_task(run_scraper_script, new_task.id)
    
    return {
        "status": "success",
        "task_id": new_task.id,
        "message": "Proses crawling website Cliste sedang berjalan di background.",
        "check_status_at": f"/api/v1/ingest/status"
    }


@router.get("/status")
def get_ingest_status(db: Session = Depends(get_db)):
    """Cek status terakhir proses ingestion"""
    latest_task = db.query(IngestLog).order_by(IngestLog.created_at.desc()).first()
    
    if not latest_task:
        return {"status": "no_tasks", "message": "Belum pernah ada proses ingestion."}
        
    return {
        "task_id": latest_task.id,
        "status": latest_task.status,
        "progress": {
            "total": latest_task.total_urls,
            "processed": latest_task.processed_urls,
            "failed": latest_task.failed_urls,
            "percentage": f"{(latest_task.processed_urls + latest_task.failed_urls) / latest_task.total_urls * 100:.1f}%" if latest_task.total_urls > 0 else "0%"
        },
        "last_message": latest_task.message,
        "started_at": latest_task.created_at,
        "updated_at": latest_task.updated_at
    }
