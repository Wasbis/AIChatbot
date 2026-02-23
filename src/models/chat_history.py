from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from datetime import datetime
from src.core.database import Base


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    role = Column(String)  # 'user' atau 'assistant'
    message = Column(Text)
    intent = Column(String, nullable=True)  # Menyimpan deteksi TECHNICAL/SALES/dll
    needs_human = Column(Boolean, default=False)  # Flag untuk notifikasi tim konsultan
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session_id = Column(String(255), index=True)
