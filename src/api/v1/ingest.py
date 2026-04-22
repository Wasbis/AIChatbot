from fastapi import APIRouter, BackgroundTasks
import subprocess
import sys

router = APIRouter()


def run_scraper_script():
    """Fungsi untuk menjalankan script scraper di background"""
    subprocess.run(
        ["python", "-u", "scripts/scrape_website.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


@router.post("/website")
def ingest_website(background_tasks: BackgroundTasks):
    """
    Endpoint untuk menyuruh AI membaca ulang website.
    Dijalankan di background agar API tidak timeout/nge-hang.
    """
    background_tasks.add_task(run_scraper_script)
    return {
        "message": "Proses crawling website Cliste sedang berjalan di background. Cek terminal untuk log."
    }
