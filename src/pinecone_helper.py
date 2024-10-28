# File for logging embeddings to Pinecone Cloud.
from pinecone.grpc import PineconeGRPC as Pinecone 
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from create_embeddings import download_hf_embeddings
import os

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key = PINECONE_API_KEY)

index_name = "medicoai"

def create_pinecone_index(index_name:str):
    """
    This function creates a Pinecone index with the specified name and configuration.
    Returns the Pinecone index object.
    RUN THIS METHOD ONLY ONCE
    """
    pc.create_index(
        name=index_name,
        dimension= 384, 
        metric="cosine",
        spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
    )
    return pc

def get_pinecone_docsearch(index_name:str, embeddings):
    """
    This function returns the Pinecone index object for the specified name.
    Args : 
    index_name : Name of the pinecone Index,
    embeddings : embeddings object from create_embeddings.py
    """
    docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
    )
    return docsearch

def describe_pinecone_stats(index):
    """
    Describes the stats of the Pinecone index.
    Args:
    index : Pinecone index object
    """
    return index.describe_index_stats()

def search_docs(docsearch:str,query:str):
    """
    Uses similarity search to search for a query in docs.
    Args: 
    docsearch : Pinecone index object,
    query : Query to search for.
    """
    return docsearch.similarity_search(query)

if __name__ == "__main__":
    index_name = "medicoai"
    embeddings = download_hf_embeddings()
    # pc_index = create_pinecone_index(index_name)
    pc_index = get_pinecone_docsearch(index_name, embeddings)
    # docsearch = get_pinecone_docsearch(index_name, embeddings)
    q1 = pc_index.search_docs("What is acne ? ")
    print(q1)