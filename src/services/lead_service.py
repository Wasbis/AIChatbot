import re
from sqlalchemy.orm import Session
from src.models.leads import Lead


def extract_and_save_lead(text: str, session_id: str, db: Session):
    """
    Fungsi untuk mengekstrak Nama, Email, dan Nomor HP dari teks chat,
    lalu menyimpannya ke database Leads.
    """
    # 1. Regex Email & Phone (Masih sama)
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"(?:\+62|62|0)8[0-9]{7,11}"

    # 2. Regex Baru untuk NAMA (Menangkap 1-2 kata setelah "saya", "dengan", "panggil")
    # Contoh yang tertangkap: "saya diya", "dengan abid", "nama saya budi"
    name_pattern = r"(?:saya|nama saya|dengan|panggil)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)"

    # Eksekusi pencarian
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    names = re.findall(
        name_pattern, text.lower()
    )  # lower() biar gak pusing huruf besar/kecil

    email = emails[0] if emails else None
    phone = phones[0] if phones else None
    # Rapihkan format nama (huruf awalan kapital)
    name = names[0].title() if names else None

    # Jika nemu salah satu aja (email / hp / nama), gass simpan!
    if email or phone or name:
        existing_lead = db.query(Lead).filter(Lead.session_id == session_id).first()

        if existing_lead:
            # Update data yang masih kosong
            if email and not existing_lead.email:
                existing_lead.email = email
            if phone and not existing_lead.phone:
                existing_lead.phone = phone
            if name and not existing_lead.name:
                existing_lead.name = name
        else:
            # Bikin data lead baru
            new_lead = Lead(session_id=session_id, email=email, phone=phone, name=name)
            db.add(new_lead)

        db.commit()
        return True

    return False
