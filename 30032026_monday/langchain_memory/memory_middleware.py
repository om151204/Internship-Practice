from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

SYSTEM_PROMPT = """
You are a professional AI assistant that keeps track of the conversation. 
Guidelines:
1. Maintain context using short-term memory (last 4 exchanges fully visible).
2. Retrieve older messages from long-term memory (MongoDB) when needed.
3. Answer clearly and naturally, human-like.
4. Recall previous questions and answers accurately. For example:
   User: What is 2+2?
   AI: 2+2 equals 4.
   User: What was my first question?
   AI: Your first question was "What is 2+2?", and the answer was "4".
5. If an answer is not found in any memory, respond politely indicating lack of information.
"""

llm = ChatOllama(
    model = OLLAMA_MODEL,
    temperature = 0
)

# Long-Term Memory
session_id = "user_1"
mongo_memory = MongoDBChatMessageHistory(
    connection_string = MONGO_URI,
    session_id = session_id,
    database_name = "LangChain",
    collection_name = "chat_history_middleware"
)

# Agent with Summerization Middleware
agent = create_agent(
    model = llm,
    checkpointer = None,
    system_prompt = SYSTEM_PROMPT,
    middleware = [
        SummarizationMiddleware(
            model = llm,
            trigger = ("messages",4), #  threshold at which the middleware starts the summarization process.
            keep = ("messages",4),    #  ensures that the 4 most recent messages are kept exactly as they are, while everything older is compressed into the summary.
        )
    ]
)

# Function to update MongoDB for old messages
def update_long_term(local_messages):
    # more than 4 user and 4 AI Messages
    # Store old messages to MongoDB
    if len(local_messages) > 8:
        old = local_messages[:-8]
        for msg in old:
            # if isinstance(msg.content, HumanMessage) or isinstance(msg.content, AIMessage):
            mongo_memory.add_message(msg)
        return local_messages[-8:]
    return local_messages

# Searching in Long Term Memory
def search_long_term(query):
    for msg in reversed(mongo_memory.messages):
        if query.lower() in msg.content.lower():
            return msg
    return None

# Chat Loop
def chat():
    thread_id = "thread_1"
    local_messages = []  # Start with system prompt

    while True:
        query = input("\nUser:- ")
        if query.lower() == "exit":
            break

        # Step 1: Building Prompt
        # The middleware automatically summerizes last 4 messages
        prompt_message = HumanMessage(content=query)

        # Step 2: Check long-term memory if query not in summary
        retrieved_mssg = search_long_term(query)

        if retrieved_mssg:
            prompt_message.content += f"\nRelevant old memory:{retrieved_mssg.content}"

        # Step 3: Invoke Agent
        result = agent.invoke(
            {"messages":local_messages + [prompt_message]},
            config = {"configurable": {"thread_id": thread_id}},
        )
        response = result["messages"][-1].content
        print("\nAI:- ",response)

        # Step 4: Save new exchanges
        local_messages.append(HumanMessage(content=query))
        local_messages.append(AIMessage(content=response))

        # Step 5: Trim old messages and update MongoDB
        local_messages = update_long_term(local_messages)

if __name__ == "__main__":
    chat()






