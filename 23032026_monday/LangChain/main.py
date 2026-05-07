from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Initialize model
# llm = ChatOllama(
#     model="llama3.2:latest",
#     temperature=0.7,
#     base_url=BASE_URL,
# )

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

# Chat
response = llm.invoke([
    HumanMessage(content="Explain what is deep learning")
])

print(response.content)