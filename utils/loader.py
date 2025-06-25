import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader

def load_documents(folder_path: str):
    documents = []
    if not os.path.exists(folder_path):
        st.error(f"The folder '{folder_path}' does not exist. Please create it and add PDF or TXT files.")
        return documents
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
            elif filename.endswith(".txt"):
                st.write(f"Loading text file: {filepath}")  # Debug info
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
        except Exception as e:
            st.error(f"Error loading file {filename}: {e}")
    return documents