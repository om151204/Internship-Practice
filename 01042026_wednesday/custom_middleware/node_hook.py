import os
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

# Load environment variables (like API keys or model names)
load_dotenv()

ollama_model = os.getenv("OLLAMA_MODEL")

# This 'Node-style' hook runs BEFORE the LLM is called
# can_jump_to=["end"] allows this middleware to halt execution and skip to the final result
@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime):
    print("Message count :", len(state["messages"]))

    # Logic: If the conversation history is too long (4+ messages), stop the agent
    if len(state["messages"]) >= 4:
        return {
            # Provide a final response so the user isn't left hanging
            "messages": [AIMessage("Conversation limit reached.")],
            # Trigger the jump to the 'end' node of the graph
            "jump_to": "end"
        }
    return None


# This 'Node-style' hook runs AFTER the LLM generates a response
@after_model
def log_response(state: AgentState, runtime: Runtime):
    # Log the last message added to the state (the AI's response)
    print(f"Model returned: {state['messages'][-1].content}\n")
    return None

model = ChatOllama(
    model=ollama_model,
    temperature=0.3
)

# create_agent builds the underlying graph with the middleware layers injected
agent = create_agent(
    model,
    middleware=[check_message_limit, log_response],
    # InMemorySaver persists 'thread_id' state in RAM so the agent remembers history
    checkpointer=InMemorySaver()
)

# Configuration for state persistence (useful for maintaining separate user sessions)
config = {"configurable": {"thread_id": "thread_1"}}


while True:
    user_input = input("\nEnter your message: ")
    if user_input.lower() == "exit":
        print("Conversation Ended")
        break

    inputs = {"messages": [HumanMessage(content=user_input)]}

    # invoke() triggers the agent loop (Middleware -> LLM -> Middleware)
    result = agent.invoke(inputs, config=config)

    # Print the final response from the agent's updated state
    print(result["messages"][-1].content)
