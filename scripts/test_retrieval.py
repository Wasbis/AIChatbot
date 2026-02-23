import os
import sys
from pathlib import Path

# Fix routing (Sama kayak di ingest_initial_data.py)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Arahkan ke folder database lokal kita
CHROMA_DB_DIR = "./data/vectorstore"


def test_search(query: str):
    print(f"🔍 Mencari jawaban di database untuk: '{query}'\n")

    # 1. Panggil mesin pembaca vektor yang sama persis
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Koneksikan ke database (kayak connect() di mongoose/prisma JS)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="cliste_knowledge",
    )

    # 3. Eksekusi pencarian vektor! (k=3 artinya ambil 3 hasil paling mirip)
    results = vectorstore.similarity_search(query, k=3)

    if not results:
        print("❌ Tidak ditemukan data yang relevan.")
        return

    print(f"✅ Ketemu! Ini {len(results)} potongan dokumen paling nyambung:\n")

    # 4. Looping untuk nampilin hasilnya
    for i, doc in enumerate(results):
        print(f"=== HASIL {i+1} ===")
        # doc.metadata ini JSON object biasa, kita bisa ambil source & page-nya
        print(f"Sumber  : {doc.metadata.get('source', 'Unknown')}")
        print(f"Halaman : {doc.metadata.get('page', 'Unknown')}")
        print(f"Isi Teks: {doc.page_content}\n")


if __name__ == "__main__":
    # GANTI TEKS DI BAWAH INI dengan pertanyaan yang jawabannya
    # pasti ada di dalam PDF CRI_companyprofile_2026.pdf lu.
    pertanyaan = "Apa saja layanan utama atau produk dari perusahaan ini?"

    test_search(pertanyaan)
