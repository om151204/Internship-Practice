import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

base_url = os.getenv("BASE_URL")
DB_PATH = "chromadb"

def create_vectorstore():
    """
    This function loads the documents, splits them into chunks, creates embeddings for the chunks using
    OllamaEmbeddings, and then stores the embeddings in a Chroma vector store.
    The vector store is persisted to disk at the specified DB_PATH.
    :return: None
    """
    try:
        document_1 = PyMuPDFLoader("rag_documents/company_policies.pdf")
        document_2 = PyMuPDFLoader("rag_documents/product_manual.pdf")
        document_3 = PyMuPDFLoader("rag_documents/faq.pdf")
        docs = document_1.load() + document_2.load() + document_3.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        embeddings = OllamaEmbeddings(model = "mistral-nemo", base_url = base_url)

        vectorstore = Chroma.from_documents(
            chunks, embeddings,
            persist_directory = DB_PATH
        )
        persist_directory = DB_PATH
        return persist_directory,vectorstore
    except FileNotFoundError:
        print("File not found. Please ensure that the document files are in the correct path.")
    except Exception as e:
        print("Error in create_vectorstore",e)

def get_retriever():
    """
    This function initializes the OllamaEmbeddings and Chroma vector store, and then creates a retriever tool
    using the vector store.
    :return: None
    """
    try:
        embeddings = OllamaEmbeddings(model = "mistral-nemo", base_url = base_url)
        vectorstore = Chroma(
            persist_directory = DB_PATH,
            embedding_function = embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        return create_retriever_tool(
            retriever = retriever,
            name = "Retriever",
            description = "Retrieve the most relevant documents from the knowledge base provided"
        )
    except Exception as e:
        print("Error in get_retriever",e)

directory,vectordb = create_vectorstore()


