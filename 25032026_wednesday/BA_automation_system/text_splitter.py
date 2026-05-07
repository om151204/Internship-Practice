from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
            )
        return splitter.split_text(text)
    except Exception as e:
        print("Error in splitting the data",e)
