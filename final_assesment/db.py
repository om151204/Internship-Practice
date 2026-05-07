import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv()
CONNECTION_STRING = os.getenv("MONGO_URL")
client = None
try:
    client = MongoClient(CONNECTION_STRING)
    collection = client["final_assessment"]["conversation_history"]
except ConnectionError as e:
    print("Connection Error",e)
except Exception as e:
    print("Error connecting MongoDB",e)

memory = MongoDBSaver(client=client, collection_name="conversation_history")
