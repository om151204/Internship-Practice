import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_API_MODEL_2")

llm = ChatOllama(
    model = OLLAMA_MODEL,
    temperature = 0
)

fallback_model = ChatGroq(
    model = GROQ_MODEL,
    api_key = GROQ_API_KEY,
    temperature = 0
)

agent = create_agent(
    model = llm,
    middleware = [
        ModelFallbackMiddleware(fallback_model)
    ]
)

query = {
    "messages": [
        HumanMessage(
            "Who is the Prime Minister of India?"
        )
    ]
}

response = agent.invoke(query)
print(response["messages"][-1].content)
