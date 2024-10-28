# Helper for loading and chunking datasets.
import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF Loader
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls = PyPDFLoader)
    docs = loader.load()
    return docs

# Chunking : 
def text_split(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks