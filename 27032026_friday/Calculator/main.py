import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
API_MODEL = os.getenv("GROQ_API_MODEL_2")

llm = ChatGroq(
    api_key = API_KEY,
    model = API_MODEL,
    temperature = 0.2
)

@tool(
    "Addition",
    description = "Addition of two numbers",
)
def addition(numbers: list[float]):
    """This tool returns addition of all provided numbers"""
    return sum(numbers)


tools = [addition]

agent = create_agent(
    tools = tools,
    model = llm,
    system_prompt =  "You have to behave as an expert calculator agent"
)

user_query = input("Please enter a query:- ")

query = {
    "messages":[HumanMessage(
        content = user_query
    )]
}

res = agent.invoke(query)
print(res["messages"][-1].content)