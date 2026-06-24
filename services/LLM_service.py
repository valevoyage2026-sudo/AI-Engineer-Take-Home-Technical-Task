import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
# # Change this block in your code
# llm = ChatOllama(
#     model="gemma3:4b",  # Or "qwen2.5", "mistral", whatever model you have pulled locally
#     temperature=0.1,  # Dropping down from 1.0 slightly so structural responses are more stable
# )
