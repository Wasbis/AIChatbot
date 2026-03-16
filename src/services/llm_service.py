import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama  # Pastikan sudah: pip install langchain-ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Mesin Utama (Groq Cloud - Llama 3.3 70B)
llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY"),
)

llm_local = ChatOllama(
    model="llama3:instruct",
    temperature=0.0,
)

# 3. Hybrid Logic: Coba Groq dulu, kalau gagal (Limit/Error) pindah ke Lokal
llm = llm_groq.with_fallbacks([llm_local])


def classify_intent(user_input: str) -> str:
    """Fungsi klasifikasi intent dengan dukungan fallback otomatis"""
    intent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Klasifikasikan input user ke dalam kategori: 1. TECHNICAL, 2. SALES, 3. KONSULTASI, 4. CHITCHAT. Output HANYA satu kata.",
            ),
            ("human", "{input}"),
        ]
    )

    # Menggunakan 'llm' yang sudah dipasangi fallback
    chain = intent_prompt | llm | StrOutputParser()

    try:
        return chain.invoke({"input": user_input}).strip().upper()
    except Exception as e:
        print(f"⚠️ Warning: Kedua LLM bermasalah. Error: {e}")
        return "CHITCHAT"
