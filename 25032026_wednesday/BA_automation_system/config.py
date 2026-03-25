import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model = "llama-3.3-70b-versatile",
        temperature = 0
    )
except ConnectionError as e:
    print("Error connecting to Groq",e)
except Exception as e:
    print("Error connecting to Groq",e)
