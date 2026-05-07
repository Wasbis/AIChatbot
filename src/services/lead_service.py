from sqlalchemy import Text
from sqlalchemy.orm import Session
from src.models.leads import Lead
import json
from src.services.rag_service import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def qualify_lead_with_ai(conversation_text: str) -> dict:
    """
    Menggunakan LLM untuk mengevaluasi kualitas lead berdasarkan percakapan.
    """
    prompt_text = """
    Analisis percakapan antara AI Consultant (Cherio) dan Calon Customer berikut.
    Tentukan apakah Calon Customer ini adalah "HOT", "POTENTIAL", atau "UNCERTAIN".

    Kriteria:
    - HOT: User punya project spesifik, sedang mencari solusi untuk masalah operasional (misal: mesin rusak, data berantakan, butuh sistem baru), atau sudah menjelaskan detail teknis/industri mereka (misal: tambang, migas, pabrik). Mereka menunjukkan niat serius untuk perbaikan.
    - POTENTIAL: User bertanya tentang layanan spesifik CRI, menanyakan kapabilitas tim, atau berdiskusi soal teori implementasi (misal: gimana cara implementasi RCM?) tanpa menyebutkan masalah mendesak saat ini.
    - UNCERTAIN: Hanya sapaan, chitchat umum (apa kabar, siapa kamu), atau pertanyaan yang sangat dangkal tanpa konteks bisnis/industri.

    PENTING: Jika user berdiskusi panjang lebar soal masalah di perusahaannya (walaupun belum kasih kontak), tandai sebagai HOT atau POTENTIAL. Kita ingin menangkap histori mereka.

    Berikan output dalam format JSON murni tanpa markdown:
    {{
      "quality": "HOT" | "POTENTIAL" | "UNCERTAIN",
      "project_summary": "Ringkasan singkat (max 15 kata) tentang apa yang ingin dibuat/diselesaikan user",
      "reasoning": "Alasan singkat kenapa kamu memberikan rating tersebut"
    }}

    Percakapan:
    {conversation}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"conversation": conversation_text})
        # Bersihkan jika ada backticks markdown
        clean_response = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_response)
    except Exception as e:
        print(f"⚠️ Error Qualify Lead: {e}")
        return {
            "quality": "UNCERTAIN",
            "project_summary": "Gagal menganalisis project.",
            "reasoning": str(e)
        }


def get_lead_qualification(message: str, history: list = None) -> dict:
    """
    Analisis pesan dan history untuk mendapatkan kualitas lead tanpa menyimpan ke DB.
    """
    conv_parts = []
    if history:
        for h in history:
            role = "Customer" if h.role == "user" else "Cherio (AI)"
            conv_parts.append(f"{role}: {h.content}")
    
    conv_parts.append(f"Customer: {message}")
    full_conv = "\n".join(conv_parts)
    return qualify_lead_with_ai(full_conv)


def extract_and_save_lead(message: str, session_id: str, db: Session, history: list = None, qualification: dict = None) -> bool:
    """
    Mengekstrak kontak dan mengupdate data lead jika ditemukan.
    """
    import re

    # 1. Regex Email
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]{2,}"
    
    # 2. Regex Phone
    phone_pattern = r"(\+?\d[\d\-\s()]{8,20}\d)"

    # 3. Regex Nama & Company (Label based)
    name_match = re.search(r"(?:Nama|Name)\s*:\s*(.*)", message, re.IGNORECASE)
    company_match = re.search(r"(?:Perusahaan|Company)\s*:\s*(.*)", message, re.IGNORECASE)

    emails = re.findall(email_pattern, message)
    phones = re.findall(phone_pattern, message)
    
    clean_name = name_match.group(1).strip() if name_match else None
    clean_company = company_match.group(1).strip() if company_match else None

    # Kalau chat ini gak ngandung WA/Email, langsung false!
    if not emails and not phones:
        return False

    # Bersihin nomor HP dari spasi atau strip biar rapi di DB (opsional tapi bagus)
    clean_phone = re.sub(r'[\s\-()]', '', phones[0]) if phones else None
    clean_email = emails[0] if emails else None

    # 3. Qualification (Internal)
    if not qualification and history:
        qualification = get_lead_qualification(message, history)

    existing_lead = db.query(Lead).filter(Lead.session_id == session_id).first()

    if existing_lead:
        # Update kalau user ngasih info tambahan
        if clean_email and not existing_lead.email:
            existing_lead.email = clean_email
        if clean_phone and not existing_lead.phone:
            existing_lead.phone = clean_phone
        if clean_name:
            existing_lead.name = clean_name
        if clean_company:
            existing_lead.company = clean_company
        
        # Selalu update kualifikasi terbaru jika tersedia
        if qualification:
            existing_lead.lead_quality = qualification.get("quality")
            existing_lead.project_summary = qualification.get("project_summary")
            existing_lead.qualification_reasoning = qualification.get("reasoning")
            
        db.commit()
        return True

    # Bikin data prospek baru
    new_lead = Lead(
        session_id=session_id,
        email=clean_email,
        phone=clean_phone,
        name=clean_name if clean_name else "Prospek B2B",
        company=clean_company,
        lead_quality=qualification.get("quality") if qualification else "UNCERTAIN",
        project_summary=qualification.get("project_summary") if qualification else None,
        qualification_reasoning=qualification.get("reasoning") if qualification else None,
    )
    db.add(new_lead)
    db.commit()

    return True


def check_has_contact(session_id: str, db: Session) -> bool:
    """
    Cek apakah sesi klien ini sudah pernah memberikan kontak sebelumnya.
    """
    lead = db.query(Lead).filter(Lead.session_id == session_id).first()
    return bool(lead and (lead.email or lead.phone))