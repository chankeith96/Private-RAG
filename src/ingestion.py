from langchain.document_loaders import PyPDFLoader


def load_pdf(file_path):
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    return pages
