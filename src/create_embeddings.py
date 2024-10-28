# Creates embeddings for the given text data using the HuggingFaceEmbeddings model.
import os
from load_chunk import load_pdf, text_split
from langchain_community.embeddings import HuggingFaceEmbeddings 

def download_hf_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# embeddings = download_hf_embeddings()
# print(embeddings)



if __name__ == "__main__":
    extracted_data = load_pdf(data='..//Data/')
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    test = embeddings.embed_query("hey there")
    print(test)
    print(f"Length : {len(test)}") # 384 for 384 dimensions