import uuid
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from src.schemas.chat import ChatRequest, ChatResponse
from src.services.rag_service import get_rag_chain, classify_intent
from src.core.database import get_db
from src.models.chat_history import ChatLog
from src.services.lead_service import extract_and_save_lead, check_has_contact

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
        # Cek apakah di pesan SAAT INI user ngetik kontak
        just_captured_contact = extract_and_save_lead(
            request.message, current_session_id, db
        )

        # Cek apakah database SUDAH PUNYA kontak untuk session ini
        has_contact_in_db = check_has_contact(current_session_id, db)

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
            # Kita cuma nanya kontak KALAU database belum punya kontaknya sama sekali
            if history_count > 0 and not has_contact_in_db:
                if "kontak" not in answer.lower() and "email" not in answer.lower():
                    answer += "\n\nUntuk menyusun scope of work dan penawaran yang presisi, tim expert kami siap melakukan diskusi teknis lebih mendalam. Boleh saya minta alamat email perusahaan atau nomor WhatsApp Anda?"

            # (Bonus UX) Kalau kontak BARUSAN BANGET dapet di chat ini, kasih ucapan terima kasih!
            if just_captured_contact:
                answer += "\n\n*Terima kasih! Kontak Anda telah kami simpan. Tim engineer kami akan segera menghubungi Anda.*"

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
