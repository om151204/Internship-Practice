import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents.middleware import ModelCallLimitMiddleware

load_dotenv()

OLLAMA_MODEL_ADVANCE = os.getenv("OLLAMA_MODEL_ADVANCE")

system_prompt = """
You MUST process one Prime Minister at a time.
Call the tool separately in different steps.
Do NOT combine tool calls.
"""

llm = ChatOllama(
    model = OLLAMA_MODEL_ADVANCE,
    temperature = 0
)

@tool
def get_pm_details(name: str) -> str:
    """Get achievements for a specific Prime Minister."""
    return f"Details for {name}: Lead the country during a major era."

agent = create_agent(
    model=llm,
    tools=[get_pm_details],
    system_prompt=system_prompt,
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=1,
            exit_behavior="error"
        ),
    ]
)

try:
    query = "Get me details for Modi and Nehru"
    response = agent.invoke({"messages": [HumanMessage(content = query)]})
    print(response["messages"][-1].content)
except Exception as e:
    print(f"\nSUCCESS: Middleware triggered! Error: {e}")

