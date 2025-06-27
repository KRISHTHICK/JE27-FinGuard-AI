# FinGuard AI – Fraud Risk Analysis Agent (Streamlit + Ollama + RAG)

import streamlit as st
import os
from PyPDF2 import PdfReader
import ollama
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize ChromaDB with Ollama embeddings
def init_vector_store():
    embedding_func = OllamaEmbeddingFunction(model_name="nomic-embed-text")
    client = chromadb.Client()
    collection = client.get_or_create_collection("finguard_rag", embedding_function=embedding_func)
    return collection

# Extract text from uploaded PDF
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

# Split text into chunks for RAG
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# Add documents to vector store
def add_to_vector_store(collection, chunks, source="User Complaint"):
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"doc_{source}_{i}"], metadatas=[{"source": source}])

# Query using Ollama (RAG)
def query_agent(query, collection):
    results = collection.query(query_texts=[query], n_results=5)
    context = "\n\n".join(results["documents"][0])
    prompt = f"""
You are a Financial Risk AI Agent. Based on the following context from internal documents and policies, provide a risk analysis for the query:

Context:
{context}

Query:
{query}

Answer in bullet points with risk score (Low/Medium/High) and recommended action.
"""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Streamlit UI
st.title("FinGuard AI – Financial Fraud Risk Agent")

collection = init_vector_store()

uploaded_file = st.file_uploader("📄 Upload Complaint/Document (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and indexing document..."):
        full_text = extract_pdf_text(uploaded_file)
        chunks = split_text(full_text)
        add_to_vector_store(collection, chunks, source=uploaded_file.name)
        st.success("Document processed and added to knowledge base.")

st.markdown("---")
query = st.text_input("🔍 Ask about possible fraud pattern or analysis:", "Was there a potential fraud in this case?")

if st.button("Analyze Fraud Risk") and query:
    with st.spinner("Analyzing with AI Agent..."):
        result = query_agent(query, collection)
        st.markdown("### 🧠 Risk Analysis Output")
        st.markdown(result)
