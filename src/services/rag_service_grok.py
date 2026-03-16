import os
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
CHROMA_DB_DIR = "./data/vectorstore"

# --- 1. INISIALISASI HYBRID LLM (GROQ + LOKAL) ---
llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,  # Wajib 0.0 agar output konsisten
    api_key=os.getenv("GROQ_API_KEY"),
)

llm_local = ChatOllama(
    model="llama3:instruct",
    temperature=0.0,
)

# Jika Groq kena limit (429), otomatis pindah ke lokal
llm = llm_groq.with_fallbacks(
    [llm_local],
    exceptions_to_handle=(RateLimitError, Exception)
)

# --- 2. FUNGSI KLASIFIKASI (KEMBALI KE SINI) ---
def classify_intent(user_input: str):
    intent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Klasifikasikan input user ke dalam salah satu kategori:
        1. TECHNICAL: Pertanyaan engineering/reliability mendalam.
        2. SALES: Tanya harga, durasi, proposal, atau ingin kerjasama.
        3. KONSULTASI: User ingin diskusi, minta bantuan, atau konsultasi.
        4. CHITCHAT: Sapaan (halo, pagi) atau hal umum.
        
        Output HANYA satu kata: TECHNICAL, SALES, KONSULTASI, atau CHITCHAT.""",
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
def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="cliste_knowledge",
    )

    # --- PENTING: k=12 agar halaman website spesifik punya peluang lebih besar ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    # --- THE ENGINEER-LEVEL SYSTEM PROMPT (V7 - SPECIFIC LINK ANCHORING) ---
    system_prompt = """Anda adalah "Cherio", Executive Technical Consultant di Cliste Rekayasa Indonesia (CRI).

[STATUS KONTAK KLIEN SAAT INI: {contact_status}]

[1. TONE, PERSONALITY & MULTILINGUAL]
- Wibawa & Profesional: Gaya bahasa elegan, solutif, dan *to the point*.
- Adaptif: Jawab menggunakan BAHASA YANG SAMA dengan input terakhir user. 
- ANTI-REPETISI: JANGAN PERNAH memperkenalkan diri jika sudah ada di <riwayat_chat>.

[2. FORMATTING & MARKDOWN]
- PARAGRAF: WAJIB gunakan double enter (\\n\\n) setiap 2-3 kalimat.
- LISTING: WAJIB menggunakan bullet points (- ) untuk daftar poin.
- BOLD: WAJIB gunakan **teks tebal** untuk istilah teknis atau layanan.

[3. STRICT CONSTRAINTS]
- GROUNDING: Jawab murni 100% dari tag <konteks>. Jika tidak ada, akui keterbatasan Anda.
- CEGAH AMNESIA: JANGAN tanyakan hal yang sudah dijelaskan user di <riwayat_chat>.

[4. ATURAN KONSULTASI & MEMINTA KONTAK]
- RULE 1 - JIKA STATUS [SUDAH_DAPAT]: DILARANG KERAS meminta kontak lagi.
- RULE 2 - TIMING ([BELUM_DAPAT]): HANYA boleh minta kontak jika masalah dasar sudah tergali DAN Anda tidak bisa memberikan jawaban lebih teknis lagi.
- RULE 3 - FORMAT SOFT CLOSE: Gunakan kalimat natural untuk mengajak diskusi lebih lanjut.
- RULE 4 - MAKSIMAL 1 PERTANYAAN: DILARANG memberikan lebih dari 1 pertanyaan balik.

[5. REFERENSI URL & LINK SPESIFIK (ATURAN MUTLAK!)]
- JANGAN memberikan link general (cliste.co.id) jika ada link yang lebih spesifik di dalam "SUMBER REFERENSI" (misal: /about, /services, atau halaman profil).
- Anda WAJIB mengambil URL persis yang tertera pada atribut "SUMBER REFERENSI" dari dokumen yang paling relevan dengan jawaban Anda.
- Format Wajib (Markdown): "\\n\\n🔗 **Pelajari lebih lanjut:** [Judul Halaman/Topik](URL_SPESIFIK_DARI_METADATA)"
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

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    document_prompt = PromptTemplate.from_template(
        "SUMBER REFERENSI: {source} (Halaman/Bagian: {page})\\nISI DOKUMEN:\\n{page_content}\\n"
    )

    # Chain sekarang menggunakan 'llm' hybrid otomatis
    question_answer_chain = create_stuff_documents_chain(
        llm=llm, prompt=prompt, document_prompt=document_prompt
    )

    return create_retrieval_chain(retriever, question_answer_chain)