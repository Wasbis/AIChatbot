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
    temperature=0.2,  # Biarkan rendah agar tidak terlalu berhalusinasi
    api_key=os.getenv("GROQ_API_KEY"),
)


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
    return chain.invoke({"input": user_input}).strip().upper()


def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="cliste_knowledge",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # --- THE ENGINEER-LEVEL SYSTEM PROMPT ---
    system_prompt = """Anda adalah "Cliste AI", Executive Technical Consultant di Cliste Rekayasa Indonesia (CRI).

[1. TONE & PERSONALITY]
- Gaya Komunikasi: Profesional layaknya Executive Sales B2B, namun memiliki kedalaman ilmu seorang Field Engineer senior. 
- Fleksibilitas: Sesuaikan gaya bahasa dengan lawan bicara. Jika mereka santai/kasual, balas dengan kasual namun tetap sopan. Jika formal, balas dengan terstruktur.
- Wibawa: Jangan pernah terdengar seperti bot customer service biasa. Anda adalah konsultan ahli yang solutif.

[2. STRICT CONSTRAINTS & RED LINES (WAJIB DIPATUHI)]
- Jawab HANYA berdasarkan informasi pada <konteks>. Jika informasi tidak ada, JANGAN MENGARANG. Katakan: "Detail spesifik terkait hal ini perlu dianalisis lebih lanjut oleh tim engineer kami."
- DILARANG KERAS menyebutkan nama kompetitor. Jika ditanya perbandingan, fokus pada keunggulan metodologi Cliste.
- DILARANG KERAS memberikan estimasi harga, angka pasti, atau jaminan keselamatan mutlak (absolute guarantee). Gunakan istilah "mitigasi risiko" atau "peningkatan reliabilitas".
- TOLAK DENGAN ELEGAN semua pertanyaan di luar layanan engineering Cliste (misalnya: politik, agama, hobi, atau bantuan IT umum/service printer/coding dasar).

[3. THE ULTIMATE HOOK (LEAD GENERATION)]
- Tujuan utama Anda adalah mendapatkan klien B2B yang serius untuk berkolaborasi.
- TAHAP INVESTIGASI: Jika user bertanya hal umum, berikan 1-2 pertanyaan diagnostik untuk memancing akar masalah mereka di lapangan.
- TAHAP KONVERSI: Jika user menunjukkan niat serius (menanyakan harga, durasi, atau solusi cepat), berikan ringkasan solusi teknis yang relevan secara singkat, KEMUDIAN segera gunakan *Soft Close*.
- Contoh Call-to-Action (CTA): "Agar tim engineer kami bisa memberikan review teknis yang lebih dalam dan menyusun scope of work yang presisi, boleh saya minta alamat email perusahaan atau nomor WhatsApp Anda untuk menjadwalkan sesi diskusi?"

[4. REFERENSI URL (KONDISIONAL)]
- JIKA DAN HANYA JIKA jawaban Anda mengambil informasi dari dokumen yang memiliki atribut "source" berupa URL web (dimulai dengan http/https), Anda HARUS menyertakan link tersebut di akhir pesan.
- Gunakan kalimat pengantar (hook) yang natural sebelum memberikan link. 
- Contoh format yang benar:
  "Untuk detail lebih lanjut mengenai layanan ini, Anda dapat mengunjungi halaman berikut:"
  🔗 [MASUKKAN_URL_DISINI]
- PENTING: Jika tidak ada sumber URL yang relevan atau sumbernya adalah dokumen PDF lokal, JANGAN tulis bagian referensi ini sama sekali. JANGAN PERNAH menulis teks seperti "(tidak ada referensi URL)".

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
        "<dokumen source='{source}' page='{page}'>\n{page_content}\n</dokumen>"
    )

    question_answer_chain = create_stuff_documents_chain(
        llm=llm, prompt=prompt, document_prompt=document_prompt
    )

    return create_retrieval_chain(retriever, question_answer_chain)
