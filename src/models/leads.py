from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from src.core.database import Base


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    name = Column(String, nullable=True)
    session_id = Column(String(255), index=True)
    project_summary = Column(Text, nullable=True)
    lead_quality = Column(String, nullable=True)  # HOT, POTENTIAL, UNCERTAIN
    qualification_reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
