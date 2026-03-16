from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.core.database import get_db

# Pastikan model Lead lu udah ada di src/models/leads.py
from src.models.leads import Lead

router = APIRouter()


@router.get("/")
def get_all_leads(db: Session = Depends(get_db)):
    """Mengambil semua data leads (kontak) yang berhasil ditangkap AI"""
    leads = db.query(Lead).order_by(Lead.created_at.desc()).all()
    return {"status": "success", "total_leads": len(leads), "data": leads}
