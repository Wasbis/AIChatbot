import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForImageTextToText, pipeline, BitsAndBytesConfig
import torch
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
CHROMA_DB_DIR = "./data/vectorstore"

# --- 1. INISIALISASI LLM (LOKAL VIA TRANSFORMERS) ---
repo_id = "google/gemma-4-E2B-it"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
os.environ["HF_TOKEN"] = hf_token  # Set env HF_TOKEN global biar library huggingface_hub nggak bawel nanya token

print("=" * 60)
print("[STARTUP] Inisialisasi Gemma 4 E2B-it...")
print("=" * 60)

print("\n[1/4] Memuat Tokenizer dari cache lokal...")
tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token, local_files_only=True)
print("      [OK] Tokenizer siap!")

print("\n[1.5/4] 🗜️  Setting up 4-bit Quantization (Supaya RAM laptop lu nggak jebol)...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("\n[2/4] Memuat bobot model (~2.5GB setelah di-press 4-bit) ke VRAM RTX 3060...")
print("      Ini tahap paling lama (estimasi 1-3 menit)...")
model = AutoModelForImageTextToText.from_pretrained(
    repo_id,
    token=hf_token,
    device_map={"": 0},
    quantization_config=quant_config,
    low_cpu_mem_usage=True,
    local_files_only=True  # KITA KUNCI LAGI JADI OFFLINE MURNI 100%
)
print(f"      [OK] Model siap! Device: {next(model.parameters()).device}")

print("\n[3/4] Merakit inference pipeline & CUDA kernels...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.4,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False
)
print("      [OK] Pipeline siap!")

print("\n[4/4] Wrapping ke LangChain interface...")
llm_local_pipeline = HuggingFacePipeline(pipeline=pipe)
llm = ChatHuggingFace(llm=llm_local_pipeline)
print("      [OK] LLM siap dirantai ke RAG chain!")

print("\n" + "=" * 60)
print("[STARTUP COMPLETE] Cherio AI siap melayani.")
print("=" * 60 + "\n")

# --- 2. FUNGSI KLASIFIKASI ---
def classify_intent(user_input: str):
    intent_prompt = ChatPromptTemplate.from_template(
        """Klasifikasikan input user ke dalam salah satu kategori:
        1. TECHNICAL: Pertanyaan engineering/reliability mendalam.
        2. SALES: Tanya harga, durasi, proposal, atau ingin kerjasama.
        3. KONSULTASI: User ingin diskusi, minta bantuan, atau konsultasi.
        4. CHITCHAT: Sapaan (halo, pagi) atau hal umum.
        
        Output HANYA satu kata: TECHNICAL, SALES, KONSULTASI, atau CHITCHAT.
        
        Input: {input}"""
    )
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

    # --- PROMPT V14: 2B-OPTIMIZED MICRO PROMPT ---
    # Model 2B tidak punya peran "system" asli, jadi Langchain akan menjejalkan ini ke awal teks "user".
    # Supaya 2B nggak lupa perannya pas ketemu teks yang panjang, kita letakkan perintah identitas 
    # TEPAT SEBELUM input user (Bottom-Heavy / Recency Bias Injection).
    
    system_prompt = """Informasi Perusahaan dan Layanan Cliste:
{context}

Riwayat Chat kita sebelumnya:
{chat_history}"""

    # PERBAIKAN: Posisikan sebagai konsultan bunglon (Mirroring). Sopan, tapi bisa santai kalau diajak santai.
    human_instruction = """Berperanlah sebagai Cherio, Technical Consultant di Cliste (CRI).
Ikuti aturan gaya bahasa berikut:
1. Bersikaplah sopan, profesional, dan solutif. Dilarang bersikap kasar atau tidak sopan.
2. SESUAIKAN BAHASA: Jawab menggunakan bahasa yang sama dengan pertanyaanku (bisa Bahasa Indonesia atau English).
3. SESUAIKAN NADA: Jika kalimatku formal, jawab dengan formal (Anda/Saya). Jika kalimatku santai, jawab dengan santai namun tetap sopan.
4. Jawab singkat berdasarkan riwayat chat dan referensi di atas. Jangan mengulang perkenalan diri.

Pertanyaanku: {input}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_instruction)
    ])

    # --- LCEL CHAIN (Kompatibel dengan LangChain v1.x+) ---
    # Langkah 1: Tarik dokumen dan passing semua variabel input secara paralel
    retrieval_step = RunnableParallel(
        context=itemgetter("input") | retriever,
        input=itemgetter("input"),
        chat_history=itemgetter("chat_history"),
        contact_status=itemgetter("contact_status")  # Wajib ada, dikirim dari chat.py
    )

    # Langkah 2: Format context jadi string, lempar ke LLM, tapi simpan context asli untuk tracing
    rag_chain = retrieval_step.assign(
        answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
            | (lambda text: text.replace("<turn|>", "").replace("<end_of_turn>", "").strip())
        )
    )

    return rag_chain

