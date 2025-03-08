from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    return embeddings