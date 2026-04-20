import re
from sqlalchemy.orm import Session
from src.models.leads import Lead


def extract_and_save_lead(message: str, session_id: str, db: Session) -> bool:
    """
    HANYA me-return True jika ada kontak BARU yang berhasil ditangkap di pesan SAAT INI.
    """
    # 1. Regex Email: Ujungnya wajib minimal 2 huruf abjad (com, id, sg), ga boleh titik.
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]{2,}"
    
    # 2. Regex Phone: Support kode negara bule (+65, +1, dll) plus spasi atau strip
    phone_pattern = r"(\+?\d[\d\-\s()]{8,20}\d)"

    emails = re.findall(email_pattern, message)
    phones = re.findall(phone_pattern, message)

    # Kalau chat ini gak ngandung WA/Email, langsung false!
    if not emails and not phones:
        return False

    # Bersihin nomor HP dari spasi atau strip biar rapi di DB (opsional tapi bagus)
    clean_phone = re.sub(r'[\s\-()]', '', phones[0]) if phones else None
    clean_email = emails[0] if emails else None

    existing_lead = db.query(Lead).filter(Lead.session_id == session_id).first()

    if existing_lead:
        # Update kalau user ngasih info tambahan
        if clean_email and not existing_lead.email:
            existing_lead.email = clean_email
        if clean_phone and not existing_lead.phone:
            existing_lead.phone = clean_phone
        db.commit()
        return True  # Kontak baru berhasil ditambahkan

    # Bikin data prospek baru
    new_lead = Lead(
        session_id=session_id,
        email=clean_email,
        phone=clean_phone,
        name="Prospek B2B",
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