import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
base = os.environ.get("BASE_URL")

# Initialize model
llm = ChatOllama(
    model="llama3.2:latest",  # or mistral, phi, etc.
    temperature=0.7,
    base_url= base
)
# Chat
response = llm.invoke([
    HumanMessage(content="Explain transformers in simple terms")
])

print(response.content)

