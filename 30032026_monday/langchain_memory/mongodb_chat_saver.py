from collections import deque
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_ollama import ChatOllama
load_dotenv()

MONGO_URI = os.getenv("MONGO_URL")
DB_NAME = "LangChain"
COLLECTION_NAME = "langchain_chatbot"
SESSION_ID = "user_1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Langchain MongoDB Memory
mongo_memory = MongoDBChatMessageHistory(
    connection_string=MONGO_URI,
    database_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    session_id=SESSION_ID
)


class ShortTermMemory:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def add_message(self, message):
        # Check if full, then pop the oldest object (not an empty string)
        # and move it to MongoDB before adding the new one.
        if len(self.memory) == self.memory.maxlen:
            oldest_msg = self.memory.popleft()
            mongo_memory.add_message(oldest_msg)

        self.memory.append(message)

    def get_messages(self):
        return list(self.memory)


# Loading Context from MongoDB
def get_long_term_context(limit=4):
    messages = mongo_memory.messages
    if not messages:
        return []
    # FIX: Use slicing [-limit:] to return a LIST of messages, not a single object
    return messages[-limit:]


# Ollama Model
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.3
)

# Main Chat Functionality
memory = ShortTermMemory(max_size=4)


def chat():
    print("Welcome to ChatBot (type 'exit' to stop)\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # 1. GET CONTEXT FIRST (Before adding the new message)
        # This ensures the LLM sees what happened BEFORE this current turn
        short_context = memory.get_messages()
        long_context = get_long_term_context()

        # 2. Combine and add the current prompt
        system_msg = [SystemMessage(content="You are a helpful assistant.")]
        human_msg = HumanMessage(content=user_input)

        # Long Term + Short Term + Current Message
        final_context = system_msg + long_context + short_context + [human_msg]

        # 3. Get response
        response = llm.invoke(final_context)
        print("\nBot:", response.content, "\n")

        # 4. NOW update memory for the NEXT turn
        memory.add_message(human_msg)
        memory.add_message(AIMessage(content=response.content))

if __name__ == "__main__":
    chat()
