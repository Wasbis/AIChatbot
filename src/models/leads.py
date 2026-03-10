from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from src.core.database import Base


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    name = Column(String, nullable=True)
    session_id = Column(String(255), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
