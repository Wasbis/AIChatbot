import uuid
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from src.schemas.chat import ChatRequest, ChatResponse
from src.services.rag_service import get_rag_chain, classify_intent
from src.services.lead_service import extract_and_save_lead  # IMPORT INTELNYA
from src.core.database import get_db
from src.models.chat_history import ChatLog

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
        # AI akan memindai pesan user (misal: "Boleh, hubungi WA gue 081234567890")
        is_contact_captured = extract_and_save_lead(
            request.message, current_session_id, db
        )
        if is_contact_captured:
            print(
                f"🎯 BINGO! Kontak berhasil ditangkap untuk session: {current_session_id}"
            )

        # 3. FORMAT HISTORY
        formatted_history = ""
        if request.history:
            for chat in request.history:
                role = "Klien" if chat.role == "user" else "AI"
                formatted_history += f"{role}: {chat.content}\n"

        # 4. JALANKAN RAG
        response = rag_chain.invoke(
            {
                "input": request.message,
                "chat_history": formatted_history or "Tidak ada riwayat.",
            }
        )

        answer = response["answer"]

        needs_human = False
        if intent in ["SALES", "KONSULTASI"]:
            needs_human = True
            history_count = len(request.history) if request.history else 0

            # --- 5. LOGIKA NGELES PROAKTIF (Update) ---
            # Kita cuma nanya kontak KALAU kontaknya belum berhasil dicapture
            if history_count > 0 and not is_contact_captured:
                # Cek ulang di string answer-nya LLM biar gak nanya 2 kali
                if "kontak" not in answer.lower() and "email" not in answer.lower():
                    answer += "\n\nUntuk diskusi teknis lebih detail, boleh saya minta nomor WhatsApp atau email Anda? Tim ahli kami akan segera menghubungi."

            # (Bonus UX) Kalau kontak barusan dapet, kasih ucapan terima kasih!
            if is_contact_captured:
                answer += "\n\n*Terima kasih atas kontaknya! Tim konsultan kami akan segera menghubungi Anda.*"

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
        raise HTTPException(status_code=500, detail=str(e))
