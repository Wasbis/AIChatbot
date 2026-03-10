from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1 import chat

# --- 1. IMPORT DATABASE & MODELS ---
# Sesuaikan path dengan struktur folder lu (src/core/database.py)
from src.core.database import engine, Base

# Import model wajib dilakukan di sini agar SQLAlchemy 'membaca' cetakannya sebelum bikin tabel
from src.models import leads, chat_history

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

# Daftarin endpoint chat tadi
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])


@app.get("/")
def read_root():
    return {"message": "API Cliste AI Consultant menyala! 🚀"}
