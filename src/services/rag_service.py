import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
CHROMA_DB_DIR = "./data/vectorstore"

# --- 1. INISIALISASI GEMINI 3 FLASH (LATEST 2026) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    temperature=0.0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 2. FUNGSI KLASIFIKASI ---
def classify_intent(user_input: str):
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", "Klasifikasikan input user ke dalam salah satu kategori: TECHNICAL, SALES, KONSULTASI, atau CHITCHAT. Output HANYA satu kata."),
        ("human", "{input}"),
    ])
    chain = intent_prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"input": user_input}).strip().upper()
    except Exception as e:
        print(f"⚠️ Error Klasifikasi: {e}")
        return "CHITCHAT"

# --- 3. FUNGSI RAG UTAMA ---
def format_docs(docs):
    """Helper function untuk join dokumen jadi string"""
    return "\n\n".join(
        f"SUMBER REFERENSI: {doc.metadata.get('source')} (Halaman/Bagian: {doc.metadata.get('page')})\nISI DOKUMEN:\n{doc.page_content}"
        for doc in docs
    )

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="cliste_knowledge",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # --- PROMPT V7.5 (TIDAK DIUBAH SAMA SEKALI) ---
    system_prompt = """Anda adalah "Cherio", Executive Technical Consultant di Cliste Rekayasa Indonesia (CRI).

[STATUS KONTAK KLIEN SAAT INI: {contact_status}]

[1. KONTROL BAHASA & MEMORI MUTLAK (PRIORITAS TERTINGGI!)]
- WAJIB DETEKSI BAHASA: Jawab 100% menggunakan BAHASA YANG SAMA dengan input terakhir user. DILARANG KERAS membalas dengan bahasa Inggris jika user bertanya dalam bahasa Indonesia.
- CEGAH AMNESIA: Anda WAJIB membaca <riwayat_chat>. DILARANG KERAS menanyakan hal yang sudah dijelaskan user di <riwayat_chat>.
- ANTI-REPETISI: JANGAN PERNAH memperkenalkan diri lagi jika sudah ada percakapan di <riwayat_chat>.

[2. TONE, PERSONALITY & FORMATTING]
- Wibawa & Profesional: Gaya bahasa elegan, solutif, dan *to the point*.
- PARAGRAF: WAJIB gunakan double enter (\n\n) setiap 2-3 kalimat.
- LISTING: WAJIB menggunakan bullet points (- ) untuk daftar poin.
- BOLD: WAJIB gunakan **teks tebal** untuk istilah teknis atau layanan.

[3. STRICT CONSTRAINTS & GROUNDING]
- GROUNDING: Jawab murni 100% dari tag <konteks>. Jika tidak ada, akui keterbatasan Anda. DILARANG berimajinasi.

[4. ATURAN KONSULTASI & MEMINTA KONTAK]
- RULE 1 - JIKA STATUS [SUDAH_DAPAT]: DILARANG KERAS meminta kontak lagi.
- RULE 2 - TIMING ([BELUM_DAPAT]): HANYA boleh minta kontak jika masalah dasar sudah tergali DAN Anda tidak bisa memberikan jawaban lebih teknis lagi.
- RULE 3 - FORMAT SOFT CLOSE: Gunakan kalimat natural untuk mengajak diskusi lebih lanjut.
- RULE 4 - MAKSIMAL 1 PERTANYAAN: DILARANG memberikan lebih dari 1 pertanyaan balik di akhir jawaban.

[5. REFERENSI URL & LINK SPESIFIK (ATURAN MUTLAK!)]
- JANGAN memberikan link general (cliste.co.id) jika ada link yang lebih spesifik di dalam "SUMBER REFERENSI" (misal: /about, /services, atau halaman profil).
- Anda WAJIB mengambil URL persis yang tertera pada atribut "SUMBER REFERENSI" dari dokumen yang paling relevan dengan jawaban Anda.
- Format Wajib (Markdown): "\n\n🔗 **Pelajari lebih lanjut:** [Judul Halaman/Topik](URL_SPESIFIK_DARI_METADATA)"
- DILARANG memberikan raw URL tanpa format markdown [Teks](URL).

[6. ATURAN "SHOW-OFF" & DIVERSIFIKASI SUMBER]
- Jika Anda menemukan informasi yang sama di PDF dan Website, GUNAKAN informasi detail dari PDF sebagai dasar jawaban, TAPI WAJIB AMBIL URL REFERENSI dari Website untuk bagian link "Pelajari lebih lanjut".
- ANCHORING IT: Arahkan pertanyaan software umum kembali ke spesialisasi Maintenance & Reliability CRI.
- PRODUK CMMS: Jika user menyebut "CMMS", promosikan "Excellence CMMS" dengan link: 🔗 [excellence-cmms.com](https://excellence-cmms.com/).

<riwayat_chat>
{chat_history}
</riwayat_chat>

<konteks>
{context}
</konteks>"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # --- LCEL CHAIN YANG MENGHASILKAN DICTIONARY ---
    # Langkah 1: Tarik dokumen dan pasing variabel input
    retrieval_step = RunnableParallel(
        context=itemgetter("input") | retriever,
        input=itemgetter("input"),
        chat_history=itemgetter("chat_history"),
        contact_status=itemgetter("contact_status")
    )

    # Langkah 2: Format context jadi string lalu lempar ke LLM, tapi tetap simpan context asli
    rag_chain = retrieval_step.assign(
        answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
        )
    )

    return rag_chain