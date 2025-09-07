# app.py
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from ollama import chat  # official Python client

load_dotenv()

# config
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "medical-chatbot")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful medical assistant. Use only the provided context when answering. If the answer is not in the context, say you don't know.")

# load embeddings + vectordb
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")  # use your existing template

@app.route("/get", methods=["POST"])
def chat_endpoint():
    user_msg = request.form.get("msg", "")
    if not user_msg:
        return "No message provided", 400

    # Retrieve top-K docs
    docs = retriever.get_relevant_documents(user_msg)

    # Build a concise context string (you can customize formatting)
    contexts = []
    for d in docs:
        src = d.metadata.get("source", d.metadata.get("chunk_id", "unknown"))
        contexts.append(f"Source: {src}\n{d.page_content}")

    combined_context = "\n\n---\n\n".join(contexts)

    # Simple safety: limit context length (characters) to avoid huge payloads
    MAX_CONTEXT_CHARS = 3000
    if len(combined_context) > MAX_CONTEXT_CHARS:
        combined_context = combined_context[:MAX_CONTEXT_CHARS]  # truncate front

    # Compose prompt for Ollama
    user_prompt = f"Context:\n{combined_context}\n\nQuestion: {user_msg}\n\nAnswer concisely and cite source if used."

    # Call Ollama via Python client (blocking)
    # The Python client chat() returns a ChatResponse; the content is under ['message']['content'] or .message.content
    try:
        response = chat(model=OLLAMA_MODEL, messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        # Extract text (support both dict-like and object-like responses)
        if isinstance(response, dict):
            answer = response.get("message", {}).get("content", "")
        else:
            # response has attribute .message.content
            answer = getattr(response, "message", {}).content if getattr(response, "message", None) else ""
            if not answer:
                # fallback to printed representation
                answer = str(response)
    except Exception as e:
        answer = f"Error from Ollama: {e}"

    return answer

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
