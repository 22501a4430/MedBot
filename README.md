# 🩺 Medical Chatbot (Local RAG with Ollama + ChromaDB)

This project is a **local medical chatbot** that can answer questions based on PDF documents (e.g., textbooks, notes, research papers).  
It uses:

- **LangChain + ChromaDB** → for storing and retrieving embeddings (local vector database)  
- **HuggingFace sentence-transformers** → to embed text chunks into vectors  
- **Ollama** → to run a local LLM (e.g., `llama3.2`) for answering questions  
- **Flask** → for a simple web interface  

✅ Works completely **offline** — no OpenAI or Pinecone required.

---

## 📦 Requirements

- **Python 3.10+**  
- **Ollama installed locally** (Windows: in `C:\Users\vijay\.ollama\ollama.exe`)  
  - Install Ollama: [https://ollama.com/download](https://ollama.com/download)  
  - Pull a model:  
    ```bash
    ollama pull <model>
    ```
- Recommended: create a virtual environment.

---

--- make sure that python virtual envirornment is running before executing the project
--- conda activate <name>

## 🔧 Installation

1. Clone this repo (or copy files into a project folder).
2. Install Python dependencies:
3. pip install -r requirements.txt

## 🏎️ execution
1. ollama run <model>
2. python store_index.py
3. python app.py
