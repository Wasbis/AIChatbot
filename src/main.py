from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import semua router dari folder api/v1
from src.api.v1 import chat, leads, ingest

# --- 1. IMPORT DATABASE & MODELS ---
from src.core.database import engine, Base

# PENTING: Kasih alias "lead_models" biar gak bentrok sama router "leads" di atas
from src.models import leads as lead_models, chat_history

# --- 2. JALANKAN MESIN PEMBUAT TABEL ---
# Baris sakti ini akan mengecek: Kalau cliste_ai.db belum ada, dia bakal bikin file dan tabelnya.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Cliste AI Consultant API",
    description="Backend API untuk Widget AI Cliste",
    version="1.0.0",
)

# Setup CORS biar frontend lu (React/Preact) nanti bisa nembak API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. DAFTARIN SEMUA ENDPOINT ---
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chatbot"])
app.include_router(leads.router, prefix="/api/v1/leads", tags=["Lead Management"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Data Ingestion"])


@app.get("/")
def read_root():
    return {"message": "API Cliste AI Consultant menyala! 🚀"}
