import asyncio
import os

# --- FIX 1: Matikan telemetri ChromaDB biar terminal bersih dari error log ---
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
CHROMA_DB_DIR = "./data/vectorstore"

# --- CONFIG ---
TARGET_SITEMAP = "https://cliste.co.id/sitemap.xml"
FALLBACK_URLS = [
    "https://cliste.co.id/",
    "https://cliste.co.id/about",
    "https://cliste.co.id/services",
]


def get_urls_from_sitemap(sitemap_url: str):
    print(f"🗺️ Mencari halaman dari Sitemap: {sitemap_url}")
    try:
        import requests

        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall(".//ns:loc", ns)]
        print(f"✅ Ditemukan {len(urls)} halaman di Sitemap!")
        return urls
    except Exception as e:
        print(f"⚠️ Gagal membaca sitemap: {e}. Menggunakan URL Fallback.")
        return FALLBACK_URLS


async def run_scraper():
    print("🕵️‍♂️ Cliste Knowledge Crawler V6 Initiated...")

    urls_to_scrape = get_urls_from_sitemap(TARGET_SITEMAP)
    all_documents = []
    failed_count = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        for i, url in enumerate(urls_to_scrape):
            print(f"\n📄 [{i+1}/{len(urls_to_scrape)}] Sedang memproses: {url}")
            try:
                # --- FIX 2: Ganti networkidle ke domcontentloaded ---
                # Ini ngasih tau browser: "Kalau teks udah muncul, sikat! Gak usah nungguin script background"
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)

                html_content = await page.content()
                soup = BeautifulSoup(html_content, "html.parser")

                page_title = soup.title.string.strip() if soup.title and soup.title.string else "Website Cliste"

                for junk in soup(
                    ["script", "style", "nav", "footer", "header", "noscript"]
                ):
                    junk.extract()

                text = soup.get_text(separator="\n")
                clean_text = "\n".join(
                    [line.strip() for line in text.splitlines() if line.strip()]
                )

                if clean_text:
                    doc = Document(
                        page_content=clean_text,
                        metadata={
                            "source": url,
                            "title": page_title,  # <-- TAMBAHKAN INI KE METADATA
                            "source_type": "WEBSITE",
                            "reliability_score": 1.2,
                            "page": f"Web_{i+1}",
                        },
                    )
                    all_documents.append(doc)
                    print(f"  ✅ Teks berhasil diekstrak! ({len(clean_text)} karakter)")
                else:
                    print("  ⚠️ Halaman kosong setelah dibersihkan.")

            except Exception as e:
                print(f"  ❌ Gagal parse URL {url}: {e}")
                failed_count += 1
                os.makedirs("logs", exist_ok=True)
                await page.screenshot(path=f"logs/error_cliste_{i}.png")

        await browser.close()

    print("\n📊 --- LAPORAN CRAWLER ---")
    print(f"Total URL Target : {len(urls_to_scrape)}")
    print(f"Berhasil Diekstrak: {len(all_documents)}")
    print(f"Gagal/Error      : {failed_count}")

    if not all_documents:
        print("🛑 Eksekusi dibatalkan, tidak ada data untuk di-ingest.")
        return

    print("\n⚙️ Memulai Ingestion ke Chroma DB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="cliste_knowledge",
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)

    print(f"🧠 Menyuntikkan {len(chunks)} potongan memori ke dalam otak AI...")
    vectorstore.add_documents(chunks)
    print("✅ Misi Selesai! AI lu sekarang hafal luar kepala isi cliste.co.id.")


if __name__ == "__main__":
    asyncio.run(run_scraper())
