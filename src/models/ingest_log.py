from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from src.core.database import Base


class IngestLog(Base):
    __tablename__ = "ingest_logs"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, default="PENDING")  # PENDING, RUNNING, SUCCESS, FAILED
    total_urls = Column(Integer, default=0)
    processed_urls = Column(Integer, default=0)
    failed_urls = Column(Integer, default=0)
    message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
