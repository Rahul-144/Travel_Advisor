from Data_loading import create_chunks
from Data_loading import process_json_files
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

processed_documents = process_json_files('Data_preparation/enwikivoyage-sectioned')
chunks = create_chunks(processed_documents)
# print(chunks)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def faiss_index():
    if os.path.exists('faiss_index'):
        print("Loading existing index")
        vectorstore = FAISS.load_local('faiss_index', embeddings,allow_dangerous_deserialization=True)
    else:
        print("Creating new index")
        vectorstore = FAISS.from_documents(chunks, embeddings,)
    return vectorstore

