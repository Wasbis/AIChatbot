from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter():
    """
    Kenapa butuh ini?
    LLM punya batas ingatan (context window). Kita gabisa masukin 1 buku langsung.
    Jadi kita potong-potong per 1000 karakter.

    chunk_overlap=150: Ini kuncinya! Kita menyisakan 150 karakter dari potongan
    sebelumnya ke potongan berikutnya supaya 'makna' atau konteks kalimatnya gak
    terputus di tengah-tengah.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
