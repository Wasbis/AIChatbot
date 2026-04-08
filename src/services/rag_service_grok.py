import os
from operator import itemgetter
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
CHROMA_DB_DIR = "./data/vectorstore"

# --- 1. INISIALISASI HYBRID LLM (GEMMA + GROQ + LOKAL) ---
# Prioritas utama: Gemma 4 E4B-it via Hugging Face Hub
llm_huggingface = HuggingFaceEndpoint(
    repo_id="google/gemma-4-E4B-it",
    task="text-generation",
    temperature=0.01,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
llm_gemma = ChatHuggingFace(llm=llm_huggingface)

llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY"),
)

llm_local = ChatOllama(
    model="llama3:instruct",
    temperature=0.0,
)

# Urutan prioritas: Gemma -> Groq -> Ollama Lokal
llm = llm_gemma.with_fallbacks(
    [llm_groq, llm_local],
    exceptions_to_handle=(RateLimitError, Exception)
)

# --- 2. FUNGSI KLASIFIKASI ---
def classify_intent(user_input: str):
    intent_prompt = ChatPromptTemplate.from_messages([
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
    ])
    chain = intent_prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"input": user_input}).strip().upper()
    except Exception as e:
        print(f"⚠️ Error Klasifikasi: {e}")
        return "CHITCHAT"

# --- 3. FUNGSI RAG UTAMA (LCEL PATTERN - KOMPATIBEL LANGCHAIN v1.x+) ---
def format_docs(docs):
    """Helper function untuk join dokumen jadi string dengan metadata."""
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

    # k=12 agar halaman website spesifik punya peluang lebih besar masuk context
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # --- LCEL CHAIN (Kompatibel dengan LangChain v1.x+) ---
    # Langkah 1: Tarik dokumen dan passing semua variabel input secara paralel
    retrieval_step = RunnableParallel(
        context=itemgetter("input") | retriever,
        input=itemgetter("input"),
        chat_history=itemgetter("chat_history"),
        contact_status=itemgetter("contact_status")
    )

    # Langkah 2: Format context jadi string, lempar ke LLM, tapi simpan context asli untuk tracing
    rag_chain = retrieval_step.assign(
        answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
        )
    )

    return rag_chain


