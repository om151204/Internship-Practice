from collections import deque
from dotenv import load_dotenv
import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_mongodb import MongoDBChatMessageHistory


load_dotenv()

MONGO_URI = os.getenv("MONGO_URL")
DB_NAME = "LangChain"
COLLECTION_NAME = "chat_history"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


class ChatState(TypedDict):
    messages: List[BaseMessage]


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.3
)


def chatbot_node(state: ChatState):
    messages = state["messages"]

    system_msg = SystemMessage(content="You are a helpful assistant.")

    response = llm.invoke([system_msg] + messages)

    return {
        "messages": messages + [AIMessage(content=response.content)]
    }

builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

class ShortTermBuffer:
    def __init__(self, max_size=4):
        self.buffer = deque(maxlen=max_size)

    def add(self, message, mongo_history):
        # If full then move oldest to MongoDB
        if len(self.buffer) == self.buffer.maxlen:
            oldest = self.buffer.popleft()
            mongo_history.add_message(oldest)

        self.buffer.append(message)

    def get(self):
        return list(self.buffer)

def chat():
    print("Chatbot (Correct FIFO MongoDB Storage)\n")

    thread_id = "user_1"
    config = {"configurable": {"thread_id": thread_id}}

    # MongoDB (long-term)
    history = MongoDBChatMessageHistory(
        connection_string=MONGO_URI,
        database_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        session_id=thread_id
    )

    # Short-term memory (last 4)
    buffer = ShortTermBuffer(max_size=4)

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        user_msg = HumanMessage(content=user_input)

        # Get context
        short_context = buffer.get()
        long_context = history.messages[-4:] if history.messages else []

        # Combine
        input_messages = long_context + short_context + [user_msg]

        # Call graph
        result = graph.invoke(
            {"messages": input_messages},
            config=config
        )

        ai_msg = result["messages"][-1]

        print(f"\nBot: {ai_msg.content}\n")

        # Add to buffer (THIS controls MongoDB storage)
        buffer.add(user_msg, history)
        buffer.add(ai_msg, history)

if __name__ == "__main__":
    chat()