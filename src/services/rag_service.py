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
    formatted = []
    for i, doc in enumerate(docs):
        source_url = doc.metadata.get('source', '')
        # Convert absolute to relative path so it works in localhost and production
        if source_url.startswith("https://websiteclistev2.cliste.id"):
            source_url = source_url.replace("https://websiteclistev2.cliste.id", "")
            if not source_url:
                source_url = "/"
                
        formatted.append(
            f"REFERENSI KE-{i+1}:\n"
            f"JUDUL HALAMAN: {doc.metadata.get('title', 'Website CRI')}\n"
            f"URL REFERENSI: {source_url}\n"
            f"ISI DOKUMEN:\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

def get_rag_chain():
    # Embedding tetep pake HuggingFace biar gak perlu ingest ulang ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="cliste_knowledge",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    system_prompt = """Anda adalah Cherio, AI Technical Consultant di Cliste Rekayasa Indonesia (CRI) — spesialis Maintenance & Reliability Excellence. Target pasar CRI adalah GLOBAL.

    STRATEGI BAHASA, KEPRIBADIAN & FORMAT (SANGAT PENTING):
    - Bicara seperti rekan kerja yang ramah, asik, cerdas, dan luwes pendengar yang baik. JANGAN KAKU SEPERTI ROBOT CS.
    - DILARANG KERAS menggunakan kata "Saya" atau "Anda", KECUALI user menggunakannya duluan. Selalu prioritaskan "Aku", "Kamu", atau "Kita".
    - Gunakan filler words natural (contoh: "Nah", "Btw", "Coba deh", "Boleh banget").
    - JANGAN pakai bullet points untuk pertanyaan umum. Gunakan paragraf yang mengalir santai.
    - Pertahankan istilah teknis dalam Bahasa Inggris (Contoh: Root Cause Analysis, Software Development) tanpa diterjemahkan.
    - FORMATTING: SELALU pisahkan kalimat penutup, pertanyaan pancingan, atau tawaran kontak di PARAGRAF BARU yang berdiri sendiri di bagian paling bawah. JANGAN digabung dengan paragraf penjelasan.

    ATURAN MEMBERIKAN LINK (HYPERLINK):
    - JIKA ADA DUA ATAU LEBIH HALAMAN RELEVAN di <konteks>, WAJIB SEBUTKAN SEMUANYA dan rangkai dalam alur obrolan yang nyambung.
    - Sisipkan link SECARA NATURAL di tengah kalimat menggunakan format markdown: **[Nama Halaman](URL)**.
    - WAJIB gunakan RELATIVE URL (misal: /services, /careers). JANGAN PERNAH gunakan absolute URL (https://...).
    - CONTOH BENAR (1 Link): "Untuk optimasi mesin, kamu bisa baca detail layanannya di halaman **[Reliability Engineering](/services/reliability-engineering)** ya."
    - CONTOH BENAR (2 Link): "Kalau mau cek posisi yang lagi buka, langsung aja mampir ke **[Career Vacancies](/careers)**. Tapi kalau nggak ada yang cocok, boleh banget ngobrol atau tanya-tanya dulu ke tim kita lewat **[Contact Us](/contactus)**!"
    - DILARANG memisahkan link menjadi daftar di akhir pesan.

    MAPPING HALAMAN UTAMA (WAJIB DIPATUHI):
    - Jika user nanya layanan secara umum, capability, industry, penawaran jasa, atau "bisa ngapain aja": ARAHKAN KE **[Our Services](/services)**.
    - Jika user nanya profil perusahaan, sejarah, atau visi misi: ARAHKAN KE **[About Us](/aboutus)**.
    - Jika user nanya loker/lowongan: ARAHKAN KE **[Career Vacancies](/careers)**.
    - JANGAN PERNAH tertukar antara halaman Services dan About Us.

    ATURAN KEJUJURAN (ANTI-HALUSINASI - SANGAT PENTING):
    1. JANGAN PERNAH MENGARANG, MENCIPTAKAN, ATAU MENEBAK JUDUL ARTIKEL/HALAMAN.
    2. Jika user bertanya tentang suatu topik (misal: RAM, RCM, RBI) dan judul artikelnya TIDAK ADA secara eksplisit di dalam <konteks>, LU WAJIB JUJUR bilang: "Wah, maaf banget, aku belum punya info atau artikel spesifik soal itu di website kita."
    3. Lu boleh menjelaskan konsep teknisnya (karena lu konsultan), TAPI jangan arahkan user ke link fiktif.

    PENGAMBILAN KONTAK [STATUS: {contact_status}] & LEAD GENERATION (CONSULTATIVE SELLING):
    - Tujuan utamamu adalah mendapatkan kontak (Email/WA), tapi JANGAN TERBURU-BURU. Gunakan pendekatan sabar ala Business Analyst.
    - TAHAP 1 (DISCOVERY): Jika user baru pertama kali menunjukkan ketertarikan (nanya harga, mau bikin software, mau maintenance project), DILARANG LANGSUNG MINTA KONTAK. Pancing mereka untuk cerita detail project, masalah, atau keluh kesahnya dulu.
      CONTOH TAHAP 1: "Wah, idenya menarik tuh! Kalau soal budget emang bervariasi tergantung kompleksitasnya. Btw, CMMS yang lama emangnya kendalanya di mana aja nih sampai mau dibikin dari nol?"
    - TAHAP 2 (CAPTURE LEAD): JIKA user SUDAH menjelaskan keluh kesahnya, masalahnya, atau gambaran project-nya, berikan empati/solusi konseptual singkat, LALU BARULAH todong kontak mereka untuk di-follow up oleh tim engineer.
      CONTOH TAHAP 2: "Oke, kebayang sekarang. Emang repot sih kalau sistem lama udah nggak support. Biar tim engineer kita bisa langsung buatin estimasi budget dan timeline yang pas, boleh bagi email atau nomor WhatsApp kamu yang aktif?"
    - JANGAN PERNAH menyuruh user pergi ke halaman Contact Us jika mereka sudah masuk tahap konsultasi ini. Kamu yang harus menjemput bola!
    - Jika STATUS = SUDAH_DAPAT: JANGAN minta kontak lagi. Lanjutkan diskusi teknis dengan santai.
    - ATURAN KERAS: Jika user mengajak meeting, Zoom, atau kunjungan kantor, kamu DILARANG mengiyakan tanpa meminta Email atau nomor WhatsApp di pesan yang sama. 
    - Kamu harus bilang: "Boleh banget! Biar tim kami bisa kirim undangan kalender dan proposal teknisnya, boleh minta Email atau nomor WhatsApp kamu yang aktif?"

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
