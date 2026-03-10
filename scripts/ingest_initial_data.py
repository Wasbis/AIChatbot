import os
import sys
from pathlib import Path

# Fix routing: Biar script di folder 'scripts' bisa baca code di folder 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.utils.chunking import get_text_splitter

# Konfigurasi Path
RAW_DATA_DIR = "./data/raw"
CHROMA_DB_DIR = "./data/vectorstore"


def ingest_documents():
    print("🚀 Memulai proses Data Pipeline (Ingestion)...")

    # 1. Pastikan folder database lokalnya ada
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    # 2. Setup Local Embedding Model
    # Note: Pas pertama kali dijalankan, ini akan download model sekitar 90MB dari internet.
    # Run berikutnya bakal instan karena modelnya udah ada di laptop lu.
    print("🧠 Menyiapkan Mesin Embedding (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Cari semua file PDF di folder data/raw
    pdf_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ Gagal: Masukin minimal 1 file PDF dulu ke folder data/raw/ !")
        return

    all_chunks = []
    text_splitter = get_text_splitter()

    # 4. Looping untuk nge-baca dan memotong setiap PDF
    for file in pdf_files:
        file_path = os.path.join(RAW_DATA_DIR, file)
        print(f"📄 Membaca dokumen: {file}")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        chunks = text_splitter.split_documents(documents)
        print(f"✂️  {file} berhasil dipecah menjadi {len(chunks)} chunks.")
        all_chunks.extend(chunks)

    # 5. Tanam ke Vector DB (ChromaDB Lokal)
    print(f"💾 Menyimpan {len(all_chunks)} chunks ke ChromaDB lokal...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="cliste_knowledge",
    )

    print(
        "✅ SUCCESS! Data lu berhasil masuk ke otak AI. Database tersimpan di data/vectorstore/"
    )


if __name__ == "__main__":
    ingest_documents()
