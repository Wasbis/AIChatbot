import uuid
import os
import logging
import json
import re  # <-- TAMBAHIN IMPORT RE DI SINI
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from src.schemas.chat import ChatRequest, ChatResponse
from src.services.rag_service import get_rag_chain, classify_intent
from src.core.database import get_db
from src.models.chat_history import ChatLog
from src.services.lead_service import extract_and_save_lead, check_has_contact
from langchain_core.messages import HumanMessage, AIMessage

# --- FUNGSI MASKING PII (SENSOR DATA SENSITIF) ---
def mask_pii(text: str) -> str:
    """Menyensor Email dan Nomor HP sebelum dikirim ke LLM/OpenAI."""
    if not text:
        return ""
    
    # 1. Regex Email
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]{2,}"
    text = re.sub(email_pattern, "[EMAIL_CENSORED]", text)
    
    # 2. Regex Phone
    phone_pattern = r"(\+?\d[\d\-\s()]{8,20}\d)"
    text = re.sub(phone_pattern, "[PHONE_CENSORED]", text)
    text = re.sub(r"\b\d{13,16}\b", "[FINANCIAL_DATA_CENSORED]", text)
    text = re.sub(r"\b\d{16}\b", "[ID_NUMBER_CENSORED]", text)
    
    return text
# -------------------------------------------------

# --- RULE-BASED INTENT PRE-FILTER (0 token, 0 LLM call) ---
_CHITCHAT_KEYWORDS = {
    "halo", "hai", "hi", "hello", "hey", "pagi", "siang", "sore", "malam",
    "apa kabar", "terima kasih", "makasih", "thanks", "thank you",
    "ok", "oke", "baik", "siap", "noted", "oke siap"
}

def classify_intent_smart(user_input: str) -> str:
    lower = user_input.lower().strip()
    if len(lower.split()) <= 5 and any(kw in lower for kw in _CHITCHAT_KEYWORDS):
        return "CHITCHAT"
    return classify_intent(user_input)

MAX_HISTORY = 4

# --- SETUP TRACING LOGGING ---
os.makedirs("logs", exist_ok=True)
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


@router.post("/")
async def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    current_session_id = request.session_id or str(uuid.uuid4())

    try:
        intent = classify_intent_smart(request.message)

        # 1. REKAM PESAN USER KE DB (SIMPAN TEKS ASLI TANPA SENSOR)
        user_log = ChatLog(
            session_id=current_session_id,
            role="user",
            message=request.message, # <-- Asli, buat histori di admin panel lu
            intent=intent,
        )
        db.add(user_log)
        db.commit()

        # --- 2. JALANKAN INTEL LEAD EXTRACTOR (AMBIL DARI TEKS ASLI) ---
        just_captured_contact = extract_and_save_lead(
            request.message, current_session_id, db, history=request.history
        )

        has_contact_in_db = check_has_contact(current_session_id, db)
        status_kontak = "SUDAH_DAPAT" if has_contact_in_db else "BELUM_DAPAT"

        if just_captured_contact:
            print(
                f"🎯 BINGO! Kontak BARU berhasil ditangkap untuk session: {current_session_id}"
            )

        # 3. FORMAT HISTORY — AMBIL HANYA N PESAN TERAKHIR (DENGAN SENSOR)
        formatted_history = []
        if request.history:
            recent = request.history[-MAX_HISTORY:]
            for chat in recent:
                # 🔥 MASKING: Sensor history chat biar LLM gak tau data sebelumnya
                masked_content = mask_pii(chat.content)
                
                if chat.role == "user":
                    formatted_history.append(HumanMessage(content=masked_content))
                else:
                    formatted_history.append(AIMessage(content=masked_content))

        # Fungsi generator untuk Server-Sent Events (SSE)
        async def generate():
            try:
                full_answer = ""
                context_docs = []

                # 🔥 MASKING: Sensor pesan user yang baru masuk sebelum dilempar ke LLM
                masked_input = mask_pii(request.message)

                # 4. JALANKAN RAG SECARA STREAMING (astream)
                async for chunk in rag_chain.astream(
                    {
                        "input": masked_input,  # <-- Pakai yang udah disensor!
                        "chat_history": formatted_history,
                        "contact_status": status_kontak,
                    }
                ):
                    if "context" in chunk and isinstance(chunk["context"], list) and not context_docs:
                        context_docs = chunk["context"]

                    if "answer" in chunk:
                        text_chunk = chunk["answer"]
                        full_answer += text_chunk
                        yield f"data: {json.dumps({'text': text_chunk, 'session_id': current_session_id})}\n\n"

                # --- 5. TULIS TRACING KE FILE LOG ---
                trace_msg = f"SESSION: {current_session_id}\n"
                trace_msg += f"👤 USER INPUT (MASKED): {masked_input}\n" # Log yg disensor biar file log jg aman
                trace_msg += f"🧠 INTENT: {intent} | CONTACT_STATE: {status_kontak}\n"
                trace_msg += f"🔍 RETRIEVED CONTEXT ({len(context_docs)} chunks):\n"

                for i, doc in enumerate(context_docs):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "-")
                    snippet = doc.page_content[:150].replace("\n", " ")
                    trace_msg += f"   [{i+1}] Source: {source} | Page: {page} -> {snippet}...\n"

                trace_msg += f"🤖 AI ANSWER: {full_answer}\n"
                trace_msg += "-" * 60

                tracer.info(trace_msg)
                # ------------------------------------

                needs_human = False
                if intent in ["SALES", "KONSULTASI"]:
                    needs_human = True

                    if just_captured_contact:
                        closing_msg = "\n\n*Terima kasih! Kontak Anda telah kami amankan. Tim engineer kami akan segera menghubungi Anda.*"
                        full_answer += closing_msg
                        yield f"data: {json.dumps({'text': closing_msg, 'session_id': current_session_id})}\n\n"

                # 6. REKAM JAWABAN AI
                ai_log = ChatLog(
                    session_id=current_session_id,
                    role="assistant",
                    message=full_answer,
                    intent=intent,
                    needs_human=needs_human,
                )
                db.add(ai_log)
                db.commit()

            except Exception as e:
                db.rollback()
                tracer.error(f"ERROR in streaming session {current_session_id}: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        db.rollback()
        tracer.error(f"ERROR in session {current_session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))