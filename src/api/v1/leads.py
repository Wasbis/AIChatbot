from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.core.database import get_db
from src.models.leads import Lead
from src.models.chat_history import ChatLog

router = APIRouter()


# =============================================
# LEADS ENDPOINTS
# =============================================

@router.get("/")
def get_all_leads(db: Session = Depends(get_db)):
    """Mengambil semua data leads (kontak) yang berhasil ditangkap AI"""
    leads = db.query(Lead).order_by(Lead.created_at.desc()).all()
    return {"status": "success", "total_leads": len(leads), "data": leads}


# =============================================
# CHAT HISTORY ENDPOINTS
# =============================================

@router.get("/history")
def get_chat_sessions(db: Session = Depends(get_db)):
    """Lihat semua session chat — ringkasan per session"""
    sessions = (
        db.query(
            ChatLog.session_id,
            func.count(ChatLog.id).label("total_messages"),
            func.min(ChatLog.created_at).label("started_at"),
            func.max(ChatLog.created_at).label("last_activity"),
            func.max(ChatLog.needs_human).label("needs_human"),
        )
        .group_by(ChatLog.session_id)
        .order_by(func.max(ChatLog.created_at).desc())
        .all()
    )

    result = []
    for s in sessions:
        # Ambil pesan pertama user sebagai preview topik
        first_msg = (
            db.query(ChatLog.message)
            .filter(ChatLog.session_id == s.session_id, ChatLog.role == "user")
            .order_by(ChatLog.created_at.asc())
            .first()
        )
        # Cek apakah session ini punya lead
        lead = db.query(Lead).filter(Lead.session_id == s.session_id).first()

        result.append({
            "session_id": s.session_id,
            "total_messages": s.total_messages,
            "started_at": str(s.started_at) if s.started_at else None,
            "last_activity": str(s.last_activity) if s.last_activity else None,
            "needs_human": bool(s.needs_human),
            "has_lead": lead is not None,
            "lead_email": lead.email if lead else None,
            "lead_phone": lead.phone if lead else None,
            "topic_preview": first_msg.message[:100] if first_msg else "—",
        })

    return {"status": "success", "total_sessions": len(result), "data": result}


@router.get("/history/{session_id}")
def get_session_detail(session_id: str, db: Session = Depends(get_db)):
    """Lihat seluruh percakapan dari 1 session tertentu"""
    messages = (
        db.query(ChatLog)
        .filter(ChatLog.session_id == session_id)
        .order_by(ChatLog.created_at.asc())
        .all()
    )

    if not messages:
        return {"status": "not_found", "message": f"Session {session_id} tidak ditemukan."}

    lead = db.query(Lead).filter(Lead.session_id == session_id).first()

    return {
        "status": "success",
        "session_id": session_id,
        "total_messages": len(messages),
        "lead": {
            "name": lead.name if lead else None,
            "email": lead.email if lead else None,
            "phone": lead.phone if lead else None,
        } if lead else None,
        "conversation": [
            {
                "role": m.role,
                "message": m.message,
                "intent": m.intent,
                "needs_human": m.needs_human,
                "timestamp": str(m.created_at),
            }
            for m in messages
        ],
    }


@router.get("/flagged")
def get_flagged_conversations(db: Session = Depends(get_db)):
    """Lihat semua percakapan yang butuh follow-up tim konsultan (needs_human=True)"""
    flagged_sessions = (
        db.query(ChatLog.session_id)
        .filter(ChatLog.needs_human == True)
        .distinct()
        .all()
    )

    result = []
    for (sid,) in flagged_sessions:
        # Ambil semua pesan dari session ini
        messages = (
            db.query(ChatLog)
            .filter(ChatLog.session_id == sid)
            .order_by(ChatLog.created_at.asc())
            .all()
        )
        lead = db.query(Lead).filter(Lead.session_id == sid).first()

        result.append({
            "session_id": sid,
            "has_lead": lead is not None,
            "lead_contact": {
                "email": lead.email if lead else None,
                "phone": lead.phone if lead else None,
            },
            "conversation": [
                {"role": m.role, "message": m.message, "timestamp": str(m.created_at)}
                for m in messages
            ],
        })

    return {"status": "success", "total_flagged": len(result), "data": result}
