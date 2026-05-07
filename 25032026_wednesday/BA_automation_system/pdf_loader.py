from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text = ""
        for page in pages:
            text += page.page_content + "\n"

        return text
    except FileNotFoundError as e:
        print("File not found, please check the file path",e)
    except Exception as e:
        print("Error loading PDF",e)