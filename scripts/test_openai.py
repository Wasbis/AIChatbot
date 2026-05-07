import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"DEBUG: API Key exists: {bool(api_key)}")
if api_key:
    print(f"DEBUG: API Key starts with: {api_key[:10]}...")

try:
    print("Testing OpenAI connection...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        max_retries=1,
        request_timeout=10.0
    )
    res = llm.invoke("Say 'Connection Successful'")
    print(f"RESULT: {res.content}")
except Exception as e:
    print(f"ERROR: {str(e)}")
