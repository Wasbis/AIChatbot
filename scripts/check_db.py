import chromadb

# 1. Buka koneksi ke folder database lokal
CHROMA_DB_DIR = "./data/vectorstore"
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# 2. Cek ada "tabel" (collection) apa aja di DB ini
collections = client.list_collections()
print(f"📂 Daftar Collection (Tabel) di DB: {collections}\n")

# 3. Akses tabel yang tadi kita isi
# Di JS ini mirip kayak db.collection('cliste_knowledge') di MongoDB
collection = client.get_collection("cliste_knowledge")

# 4. Hitung total baris data (chunks)
total_data = collection.count()
print(f"📊 Total chunks (potongan teks) yang tersimpan: {total_data}\n")

# 5. Intip (Peek) 2 data teratas untuk ngecek isinya
if total_data > 0:
    print("👀 Mengintip 2 data pertama...\n")
    # peek(2) akan mengambil 2 data teratas tanpa merubah urutan
    sample_data = collection.peek(2)

    # Response dari ChromaDB bentuknya Object (Dictionary di Python)
    # yang isinya Array (List). Kita loop biar tampilannya rapi.
    documents = sample_data.get("documents")
    ids = sample_data.get("ids")
    metadatas = sample_data.get("metadatas")
    if documents:
        for i in range(len(documents)):
            print(f"=== CHUNK KE-{i+1} ===")
            print(f"🔹 ID       : {ids[i] if ids else 'N/A'}")
            print(f"🔹 Metadata : {metadatas[i] if metadatas else 'N/A'}")
            # Kita tampilkan 300 karakter pertama aja biar terminal lu gak penuh
            print(f"🔹 Isi Teks : {documents[i][:300]}...\n")
else:
    print("⚠️ Database kosong!")
