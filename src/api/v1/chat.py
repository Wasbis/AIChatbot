import uuid
import os
import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from src.schemas.chat import ChatRequest, ChatResponse
from src.services.rag_service import get_rag_chain, classify_intent
from src.core.database import get_db
from src.models.chat_history import ChatLog
from src.services.lead_service import extract_and_save_lead, check_has_contact

# --- SETUP TRACING LOGGING ---
# Bikin folder logs kalau belum ada
os.makedirs("logs", exist_ok=True)

# Konfigurasi file log
logging.basicConfig(
    filename="logs/chat_trace.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
tracer = logging.getLogger("RAG_Tracer")
# -----------------------------

router = APIRouter()
rag_chain = get_rag_chain()


@router.post("/", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    current_session_id = request.session_id or str(uuid.uuid4())

    try:
        intent = classify_intent(request.message)

        # 1. REKAM PESAN USER
        user_log = ChatLog(
            session_id=current_session_id,
            role="user",
            message=request.message,
            intent=intent,
        )
        db.add(user_log)
        db.commit()

        # --- 2. JALANKAN INTEL LEAD EXTRACTOR ---
        just_captured_contact = extract_and_save_lead(
            request.message, current_session_id, db
        )

        has_contact_in_db = check_has_contact(current_session_id, db)
        status_kontak = "SUDAH_DAPAT" if has_contact_in_db else "BELUM_DAPAT"

        if just_captured_contact:
            print(
                f"🎯 BINGO! Kontak BARU berhasil ditangkap untuk session: {current_session_id}"
            )

        # 3. FORMAT HISTORY
        formatted_history = ""
        if request.history:
            for chat in request.history:
                role = "Klien" if chat.role == "user" else "AI"
                formatted_history += f"{role}: {chat.content}\n"

        # 4. JALANKAN RAG DENGAN STATE AWARENESS
        response = rag_chain.invoke(
            {
                "input": request.message,
                "chat_history": formatted_history or "Tidak ada riwayat.",
                "contact_status": status_kontak,
            }
        )

        answer = response["answer"]
        context_docs = response.get("context", [])  # Ambil dokumen hasil retrieval

        # --- 5. TULIS TRACING KE FILE LOG ---
        trace_msg = f"SESSION: {current_session_id}\n"
        trace_msg += f"👤 USER INPUT: {request.message}\n"
        trace_msg += f"🧠 INTENT: {intent} | CONTACT_STATE: {status_kontak}\n"
        trace_msg += f"🔍 RETRIEVED CONTEXT ({len(context_docs)} chunks):\n"

        for i, doc in enumerate(context_docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "-")
            # Potong teks konteks maksimal 150 karakter biar log gak terlalu penuh
            snippet = doc.page_content[:150].replace("\n", " ")
            trace_msg += f"   [{i+1}] Source: {source} | Page: {page} -> {snippet}...\n"

        trace_msg += f"🤖 AI ANSWER: {answer}\n"
        trace_msg += "-" * 60

        tracer.info(trace_msg)
        # ------------------------------------

        needs_human = False
        if intent in ["SALES", "KONSULTASI"]:
            needs_human = True

            if just_captured_contact:
                answer += "\n\n*Terima kasih! Kontak Anda telah kami amankan. Tim engineer kami akan segera menghubungi Anda.*"

        # 6. REKAM JAWABAN AI
        ai_log = ChatLog(
            session_id=current_session_id,
            role="assistant",
            message=answer,
            intent=intent,
            needs_human=needs_human,
        )
        db.add(ai_log)
        db.commit()

        return ChatResponse(answer=answer, session_id=current_session_id)

    except Exception as e:
        db.rollback()
        tracer.error(f"ERROR in session {current_session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))