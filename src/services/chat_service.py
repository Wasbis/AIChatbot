import sqlite3
from langchain_core.messages import HumanMessage, AIMessage
from src.services.rag_service import get_rag_chain

DB_PATH = "cliste_ai.db"


def init_db():
    """Bikin tabel history kalau belum ada di cliste_ai.db"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id: str, limit: int = 10):
    """Narik N percakapan terakhir dari database untuk jadi konteks AI"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Ambil data terbaru, lalu reverse agar urutannya kronologis (lama -> baru)
    cursor.execute(
        """
        SELECT role, content FROM (
            SELECT role, content, timestamp FROM chat_history 
            WHERE session_id = ? 
            ORDER BY timestamp DESC LIMIT ?
        ) ORDER BY timestamp ASC
    """,
        (session_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    history = []
    for role, content in rows:
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "ai":
            history.append(AIMessage(content=content))
    return history


def save_message(session_id: str, role: str, content: str):
    """Simpan pesan baru ke cliste_ai.db"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    conn.commit()
    conn.close()


def process_chat(session_id: str, user_message: str) -> str:
    """Fungsi utama yang akan dipanggil oleh API Router (FastAPI)"""
    # 1. Pastikan tabel DB ready
    init_db()

    # 2. Ambil ingatan masa lalu
    history = get_chat_history(session_id)

    # 3. Panggil RAG Chain yang udah dibikin di rag_service.py
    rag_chain = get_rag_chain()

    # 4. Eksekusi chain DENGAN history (Obat Amnesia)
    response = rag_chain.invoke({"input": user_message, "chat_history": history})

    answer = response["answer"]

    # 5. Simpan obrolan baru ke database
    save_message(session_id, "user", user_message)
    save_message(session_id, "ai", answer)

    return answer
