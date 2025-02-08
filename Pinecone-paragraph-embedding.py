# pip install "pinecone[grpc]"      # gRPC submodule more performant than HTTP 
# pip install sentence-transformers

import os
import re
from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINCONE_INDEX")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Pre-trained model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def read_chunks_from_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        paragraphs = file.read().split("\n")
    
    clean_paragraphs = [
        # Handle quotes at the beginning and end of the paragraph
        re.sub(r'^[“”"\']+|[“”"\']+$', '', paragraph.strip())
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    return clean_paragraphs

def create_and_upload_embeddings(input_file, index_name=PINECONE_INDEX):
    paragraphs = read_chunks_from_file(input_file)
    
    embeddings = model.encode(paragraphs)
    
    if pc.describe_index(index_name).name != index_name:
        print(f"Error: Index {index_name} not found")
    else:
        print(f"Found existing Pinecone index: {index_name}")

        index = pc.Index(index_name)
    
        # Prepare the embeddings
        vectors = [
            {
                "id": f"chunk_{i}",                     # Unique ID for the vector
                "values": embedding.tolist(),           # Embedding vector
                "metadata": {"text": paragraphs[i]}     # Original paragraph text as metadata
            }
            for i, embedding in enumerate(embeddings)
        ]
    
        index.upsert(vectors)
        print(f"Uploaded {len(paragraphs)} embeddings to Pinecone.")

# Example usage:
input_file = './Documents/chunked/paragraphs-chunked.txt'
create_and_upload_embeddings(input_file)
