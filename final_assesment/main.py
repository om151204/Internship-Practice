import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from rag import get_retriever
from db import memory
from langchain_ollama.chat_models import ChatOllama
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
MODEL = "mistral-nemo"

llm = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=0.1
)

tool = get_retriever()

middleware = [
    SummarizationMiddleware(
        model=llm,
        trigger = ("messages",6),
        keep = ("messages",6),
    ),
]

agent = create_agent(
    model = llm,
    middleware = middleware,
    tools = [tool],
    system_prompt="""
    You are a helpful chatbot assistant and your task is to assist the user in answering their queries 
    using the tools available to you. always use the tool when the user asks a question related to the 
    company policies, product manual or faq. if you do not know the answer do not try to hallucinate.
    """,
    checkpointer = memory
)

session_id = "User_1"
print("WELCOME TO THE CHATBOT:- (Type quit to exit)\n")
config = {"configurable": {"thread_id":"thread_1"}}

while True:
    user_query = input("Enter your query: ")
    if user_query.lower() == "quit":
        print("Exiting the chatbot. Goodbye!")
        break

    response = agent.invoke(
    {"messages": [{"role": "user", "content": user_query}]},
        session_id=session_id, config=config)

    answer = response['messages'][-1].content
    print("Chatbot:- ",answer,"\n")
