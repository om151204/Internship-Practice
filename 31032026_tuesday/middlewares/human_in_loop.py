from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langgraph.types import Command
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROOQ_API_MODEL = os.getenv("GROQ_API_MODEL_2")

llm = ChatGroq(
    api_key = GROQ_API_KEY,
    model = GROOQ_API_MODEL,
    temperature = 0
)

system_prompt = """
    You are a financial assistant.

    Rules:
    1. ALWAYS use tools when the query involves stock prices or buying shares.
    2. NEVER answer from your own knowledge for these actions.
    3. You must call the appropriate tool:
       - check_share_price → for price queries
       - buy_share → for purchase requests
    4. Return ONLY the tool call when required.
    """

@tool(
    "check_share_price",
    description = "This function is used to check share price"
)
def check_share_price(stock: str) -> str:
    """function to check share price."""
    return f"The current price of the {stock} is 300.12"

@tool(
    "buy_share",
    description = "This function is used to buy shares."
)
def buy_share(share:str) -> str:
    """function to buy shares."""
    return f"Successfully bought 10 quantity of {share}"


agent = create_agent(
    model = llm,
    tools = [check_share_price, buy_share],
    checkpointer = InMemorySaver(),
    system_prompt = system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on = {
                "buy_share": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "check_share_price": False,
            }
        )
    ]
)

config = {"configurable":{"thread_id":"thread_1"}}

while True:
    user_query = input("\nEnter your query: ")
    if user_query.lower() == "exit":
        break

    # 1. Run the agent
    state = agent.invoke(
        {"messages": [HumanMessage(content=user_query)]},
        config=config,
    )

    # 2. CHECK: Is the agent waiting for human approval?
    # We check if 'interrupts' exists in the current state
    if hasattr(state, "interrupts") and state.interrupts:
        print(f"\n--- INTERRUPT: Approval needed for {state.interrupts[0].action_request.name} ---")

        # 3. Only resume if there was an interrupt
        state = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )

    # 4. Print the final message from the assistant
    print(f"\nResponse: {state['messages'][-1].content}\n")
