import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
CHROMA_DB_DIR = "./data/vectorstore"

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)


def classify_intent(user_input: str):
    """
    IMPLEMENTASI TDD SECTION 1.1: Router / Intent Classifier
    Output ditingkatkan untuk mendeteksi KONSULTASI secara spesifik.
    """
    intent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Klasifikasikan input user ke dalam salah satu kategori:
        1. TECHNICAL: Pertanyaan engineering/reliability mendalam.
        2. SALES: Tanya harga, durasi, atau portofolio.
        3. KONSULTASI: User ingin diskusi, minta bantuan, atau konsultasi.
        4. CHITCHAT: Sapaan (halo, pagi) atau hal umum.
        
        Output HANYA satu kata: TECHNICAL, SALES, KONSULTASI, atau CHITCHAT.""",
            ),
            ("human", "{input}"),
        ]
    )

    chain = intent_prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_input}).strip().upper()


def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="cliste_knowledge",
    )
    # UPDATE: k=8 agar artikel website dan PDF bisa masuk semua ke memori
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # PROMPT BARU: Mode Agresif Lead Generation + Format Referensi Link
    system_prompt = """Anda adalah AI Consultant teknis dari Cliste Rekayasa Indonesia (CRI). 
Tugas Anda adalah mendiagnosa kebutuhan user terlebih dahulu sebelum menawarkan bantuan tim konsultan senior.

[FORMAT JAWABAN WAJIB UNTUK REFERENSI URL]
Periksa atribut "source" pada tag <dokumen> di bagian <konteks>. Jika informasi yang Anda ambil berasal dari dokumen dengan "source" berupa URL (seperti https://cliste.co.id/...), Anda WAJIB menambahkan referensi link tersebut di BARIS PALING BAWAH balasan Anda.
Gunakan format persis seperti ini di akhir pesan (pisahkan dengan enter):

🔗 Selengkapnya: [MASUKKAN_URL_DARI_ATRIBUT_SOURCE_DISINI]

[ALUR KERJA CONSULTANT]
1. TAHAP INVESTIGASI (Pesan Awal): 
   Jika user ingin konsultasi atau bertanya hal umum, jangan langsung minta kontak. 
   Tanyakan balik 1-2 pertanyaan diagnostik untuk memahami masalah mereka. 
   Contoh: "Bidang apa yang ingin Anda bahas? Apakah terkait kehandalan pabrik (Reliability), integritas aset, atau pelatihan tim?"

2. TAHAP ANALISIS: 
   Gunakan data dari <konteks> untuk memberikan gambaran solusi awal. Tunjukkan bahwa CRI ahli di bidang tersebut.

3. TAHAP KONVERSI (Jemput Bola):
   HANYA minta nomor WhatsApp/Email JIKA:
   - User bertanya soal harga atau durasi pengerjaan.
   - User bertanya hal teknis yang sangat spesifik dan tidak ada di <konteks>.
   - Percakapan sudah berlangsung lebih dari 2-3 kali dan masalah mulai mengerucut.

[ATURAN PENTING]
- Jangan beri email info@cliste.co.id di awal. Gunakan kalimat: "Agar tim engineer kami bisa memberikan review teknis yang lebih dalam..."
- Tetap ramah, profesional, dan gunakan bahasa Indonesia yang luwes (Bapak/Ibu atau Anda).

<riwayat_chat>
{chat_history}
</riwayat_chat>

<konteks>
{context}
</konteks>"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # ... (sisa code document_prompt dan chain tetap sama)
    document_prompt = PromptTemplate.from_template(
        "<dokumen source='{source}' page='{page}'>\n{page_content}\n</dokumen>"
    )

    question_answer_chain = create_stuff_documents_chain(
        llm=llm, prompt=prompt, document_prompt=document_prompt
    )

    return create_retrieval_chain(retriever, question_answer_chain)
