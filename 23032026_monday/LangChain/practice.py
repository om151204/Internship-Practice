from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader('data.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

splits = text_splitter.split_documents(documents)

print(f"Original File loaded as {len(documents)} documents")
print(f"\nAfter Splitting {len(splits)} documents\n")

for i,chunk in enumerate(splits):
    print(f"--- Chunk {i + 1} ---")
    print(chunk.page_content)  # The actual text
    print(f"Metadata: {chunk.metadata}\n")  # Info about the source