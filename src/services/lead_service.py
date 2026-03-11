import re
from sqlalchemy.orm import Session
from src.models.leads import Lead


def extract_and_save_lead(message: str, session_id: str, db: Session) -> bool:
    """
    HANYA me-return True jika ada kontak BARU yang berhasil ditangkap di pesan SAAT INI.
    """
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"(?:\+62|62|0)8[1-9][0-9]{7,11}"

    emails = re.findall(email_pattern, message)
    phones = re.findall(phone_pattern, message)

    # Kalau chat ini gak ngandung WA/Email, langsung false!
    if not emails and not phones:
        return False

    existing_lead = db.query(Lead).filter(Lead.session_id == session_id).first()

    if existing_lead:
        # Update kalau user ngasih info tambahan
        if emails and not existing_lead.email:
            existing_lead.email = emails[0]
        if phones and not existing_lead.phone:
            existing_lead.phone = phones[0]
        db.commit()
        return True  # Kontak baru berhasil ditambahkan

    # Bikin data prospek baru
    new_lead = Lead(
        session_id=session_id,
        email=emails[0] if emails else None,
        phone=phones[0] if phones else None,
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
