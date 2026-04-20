import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # <-- Berubah ke OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
CHROMA_DB_DIR = "./data/vectorstore"

# --- 1. INISIALISASI OPENAI (Lebih stabil & gak makan VRAM) ---
llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY")
)


# --- 2. FUNGSI KLASIFIKASI (Cukup di sini aja, hapus yang di file lain) ---
def classify_intent(user_input: str):
    intent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Klasifikasikan input user ke dalam salah satu kategori: TECHNICAL, SALES, KONSULTASI, atau CHITCHAT. Output HANYA satu kata.",
            ),
            ("human", "{input}"),
        ]
    )
    chain = intent_prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"input": user_input}).strip().upper()
    except Exception as e:
        print(f"⚠️ Error Klasifikasi: {e}")
        return "CHITCHAT"


# --- 3. FUNGSI RAG UTAMA ---
def format_docs(docs):
    return "\n\n".join(
        f"REFERENSI KE-{i+1}:\n"
        f"JUDUL HALAMAN: {doc.metadata.get('title', 'Website CRI')}\n"
        f"URL REFERENSI: {doc.metadata.get('source')}\n"
        f"ISI DOKUMEN:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

def get_rag_chain():
    # Embedding tetep pake HuggingFace biar gak perlu ingest ulang ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="cliste_knowledge",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    system_prompt = """Anda adalah Cherio, AI Technical Consultant di Cliste Rekayasa Indonesia (CRI) — spesialis Maintenance & Reliability Excellence.

    STRATEGI BAHASA & KEPRIBADIAN (SANGAT PENTING):
    - Bicara seperti rekan kerja yang ramah, asik, dan luwes. JANGAN KAKU SEPERTI ROBOT CS.
    - DILARANG KERAS menggunakan kata "Saya" atau "Anda", KECUALI user menggunakannya duluan. Selalu prioritaskan "Aku", "Kamu", atau "Kita".
    - Gunakan filler words natural (contoh: "Nah", "Btw", "Coba deh", "Boleh banget").
    - JANGAN pakai bullet points untuk pertanyaan umum. Gunakan paragraf yang mengalir santai.
    - Pertahankan istilah teknis dalam Bahasa Inggris (Contoh: Root Cause Analysis, Software Development) tanpa diterjemahkan.

    ATURAN MEMBERIKAN LINK (HYPERLINK):
    - JIKA ADA DUA ATAU LEBIH HALAMAN RELEVAN di <konteks> (Misal: halaman Karir DAN halaman Kontak), WAJIB SEBUTKAN SEMUANYA dan rangkai dalam alur obrolan yang nyambung.
    - Sisipkan link SECARA NATURAL di tengah kalimat menggunakan format markdown: **[Nama Halaman](URL)**.
    - CONTOH BENAR (1 Link): "Untuk optimasi mesin, kamu bisa baca detail layanannya di halaman **[Reliability Engineering](https://...)** ya."
    - CONTOH BENAR (2 Link): "Kalau mau cek posisi yang lagi buka, langsung aja mampir ke **[Career Vacancies](https://...)**. Tapi kalau nggak ada yang cocok, boleh banget ngobrol atau tanya-tanya dulu ke tim kita lewat **[Contact Us](https://...)**!"
    - DILARANG memisahkan link menjadi daftar di akhir pesan.

    PENGAMBILAN KONTAK [STATUS: {contact_status}]:
    - Jika SUDAH_DAPAT: JANGAN minta kontak lagi.
    - Jika BELUM_DAPAT: Tawarkan kontak di akhir pesan jika relevan.

    <konteks>
    {context}
    </konteks>"""
        
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    retrieval_step = RunnableParallel(
        context=itemgetter("input") | retriever,
        input=itemgetter("input"),
        chat_history=itemgetter("chat_history"),
        contact_status=itemgetter("contact_status"),
    )

    rag_chain = retrieval_step.assign(
        answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
        )
    )

    return rag_chain
