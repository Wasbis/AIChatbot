from pydantic import BaseModel
from typing import List, Optional


class ChatMessage(BaseModel):
    role: str  # isinya 'user' atau 'assistant'
    content: str  # isi pesannya


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = (
        []
    )  # Frontend akan ngirim array obrolan lama ke sini
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
