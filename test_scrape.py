import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from markdownify import markdownify as md

async def test_scrape():
    url = "https://websiteclistev2.cliste.id/ourinsight/3-common-asset-recording-mistakes-in-oil-and-gas-plants-and-their-solutions"
    print(f"Testing scrape on: {url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        await page.goto(url, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(2000)
        
        html_content = await page.content()
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Simulating the exact scraping logic
        for junk in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            junk.extract()
            
        main_content = soup.find("article") or soup.find("main") or soup.body
        if main_content:
            clean_text = md(str(main_content), strip=['a', 'img'], heading_style="ATX").strip()
        else:
            clean_text = ""
        
        print("\n--- EXTRACTED MARKDOWN ---")
        print(clean_text[:1000]) # Print first 1000 characters
        print("...\n")
        print(f"Total characters: {len(clean_text)}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_scrape())
