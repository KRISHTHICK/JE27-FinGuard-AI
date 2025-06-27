# JE27-FinGuard-AI
GenAI

💼📊 FinGuard AI – AI Agent for Financial Fraud Risk Analysis
🔍 Overview
An intelligent agent system that helps banks/insurance companies detect potential fraud by:

Collecting unstructured user complaint documents, chat logs, and emails.

Extracting key entities (name, account, transaction, timestamp, location, fraud trigger).

Using RAG (Retrieval-Augmented Generation) to provide risk explanations and action steps from internal policy docs and public fraud case data.

Deploying a conversational interface (via Streamlit + Ollama or API) for analysts to interact and query specific fraud types.

⚙️ Key Features
🧠 AI Agent Workflow:

Accepts complaint files (PDF, email, text).

Extracts entities using spaCy/LLMs.

Fetches related policies or similar fraud patterns using RAG.

Provides summary + recommended actions.


🧱 Stack
Frontend: Streamlit (UI with file upload, filters, results)

Backend: Python, FastAPI (optional)

LLM: Ollama local or Gemini API

Embedding: FAISS or ChromaDB

Document Store: Internal + Web scraped data

🚀 How to Run
bash
Copy
Edit
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Start Ollama and pull model
ollama pull llama3

# Step 3: Run the app
streamlit run app.py
