from .Data_loading import create_chunks, process_json_files
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os


def faiss_index():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists("faiss_index"):
        print("Loading existing index")

        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:
        print("Creating new index")

        processed_documents = process_json_files(
            "Data_preparation/enwikivoyage-sectioned"
        )

        chunks = create_chunks(processed_documents)

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        vectorstore.save_local("faiss_index")

    return vectorstore
