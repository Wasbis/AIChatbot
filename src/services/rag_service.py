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



CHROMA_DB_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vectorstore")

# --- 1. INISIALISASI OPENAI ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("❌ WARNING: OPENAI_API_KEY tidak ditemukan di environment variables!")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=openai_api_key,
    max_retries=3,
    request_timeout=30.0,
)


# --- 2. FUNGSI KLASIFIKASI (Cukup di sini aja, hapus yang di file lain) ---
def classify_intent(user_input: str):
    intent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Klasifikasikan input user ke dalam salah satu kategori:\n"
                "- TECHNICAL: Pertanyaan murni teori/teknis tanpa indikasi kebutuhan bisnis.\n"
                "- SALES: Menanyakan harga, cara kerjasama, atau minta meeting.\n"
                "- KONSULTASI: Menceritakan masalah di perusahaan/pabrik, minta solusi/saran untuk project nyata, atau diskusi mendalam soal implementasi layanan CRI.\n"
                "- CHITCHAT: Sapaan, basa-basi, atau hal di luar konteks CRI.\n"
                "Output HANYA satu kata: TECHNICAL, SALES, KONSULTASI, atau CHITCHAT."
            ),
            ("human", "{input}"),
        ]
    )
    chain = intent_prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"input": user_input}).strip().upper()
    except Exception as e:
        print(f"⚠️ Error Klasifikasi: {str(e)}")
        if "Connection error" in str(e):
            print("💡 Saran: Cek koneksi internet server atau limitasi firewall ke api.openai.com")
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

    ATURAN KEJUJURAN & BATASAN DOMAIN (SANGAT PENTING):
    1. JIKA user bertanya tentang hal yang benar-benar tidak relevan (misal: resep masakan, politik, hiburan), KAMU WAJIB MENOLAK UNTUK MENJAWAB SECARA SOPAN.
    2. SOFTWARE & DIGITAL TRANSFORMATION: Jika user bertanya tentang pembuatan aplikasi, website, atau software, JANGAN LANGSUNG MENOLAK. Tanyakan dulu konteksnya. Jika aplikasinya untuk industri (O&G, Pabrik, Pertambangan) atau untuk operasional (Monitoring, Asset Management, Dashboard), itu adalah kapabilitas utama CRI dalam hal Digital Transformation.
    3. JANGAN PERNAH MENGARANG, MENCIPTAKAN, ATAU MENEBAK JUDUL ARTIKEL/HALAMAN.
    4. Jika user bertanya tentang topik relevan tapi artikelnya tidak ada di <konteks>, LU WAJIB JUJUR bilang: "Wah, maaf banget, aku belum punya info spesifik soal itu di website kita." Namun, kamu boleh tawarkan untuk mendiskusikannya dengan tim engineer CRI.

    PENGAMBILAN KONTAK [STATUS: {contact_status}] & LEAD GENERATION:
    1. PRIORITAS EKSPLORASI (Gali Masalah): Jika user serius (ada project/masalah nyata), JANGAN langsung jualan form. Gali dulu detail teknisnya (misal: "Boleh tahu merk sensornya?", "Jumlah unitnya berapa?", "Sistem IT-nya pakai apa?"). Tunjukkan bahwa kamu ahli (Subject Matter Expert).
    2. JEMPUT BOLA (Soft Ask): Setelah kamu memberikan jawaban teknis yang memuaskan dan menggali info, ajak mereka isi form di layar chat SEBAGAI LANGKAH LANJUTAN. 
    3. ATURAN FREKUENSI: JANGAN meminta isi form di setiap bubble chat berturut-turut. Jika kamu sudah minta di pesan sebelumnya, fokuslah berdiskusi teknis saja di pesan sekarang.
    5. STRATEGI LEAD MAGNET (Summary): Jika user minta ringkasan/summary untuk ditunjukkan ke manajemen, berikan ringkasan singkat (teaser) saja di chat. Katakan bahwa kamu memiliki versi "Dokumen Ringkasan Profesional & Proposal Awal" yang lebih lengkap untuk dikirimkan via Email/WA. Ajak user isi form chat agar tim CRI bisa segera mengirimkan dokumen tersebut.
    6. PENGECUALIAN (Halaman Contact Us): Hanya jika user memaksa ("Gue mau hubungin langsung!") atau menolak isi form di chat, BARULAH berikan token `[CONTACT_US]`. Paragraf tepat sebelum token tersebut WAJIB diawali dengan "Ringkasan Kebutuhan:" dan berisi ANALISIS MENDALAM (Masalah inti user + Solusi CRI yang relevan + Scope project jika ada). DILARANG menulis kalimat generik seperti "ingin diskusi lebih lanjut" atau "minta dihubungi". Paragraf ini adalah memo teknis dari kamu untuk tim CRI agar mereka bisa gercep memberikan solusi.
    7. Jika STATUS = SUDAH_DAPAT: JANGAN minta kontak lagi. Fokus ke diskusi teknis.

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
