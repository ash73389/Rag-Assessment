# Rag-Assessment
# Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to query a PDF document and receive accurate answers based on its content. It is built using FastAPI, LangChain, FAISS, and a local Mistral-7B model, providing an efficient and scalable solution for document-based question answering.

The system processes the document by converting its text into vector embeddings, storing them in a FAISS database, retrieving relevant sections based on the user’s question, and generating an AI-powered response using a local LLM. It also includes a REST API and a Gradio UI for easy interaction.

How It Works
1. Document Loading & Preprocessing
The system begins by loading a PDF document using PyPDFLoader from LangChain. Since processing large documents in a single pass is inefficient, the text is split into smaller chunks using RecursiveCharacterTextSplitter. This ensures that relevant portions of the document can be retrieved without needing to process the entire text at once.

Library Used: PyPDFLoader, RecursiveCharacterTextSplitter
Chunking Strategy:
Each chunk contains 1000 characters with an overlap of 200 characters
Helps maintain context when splitting text
Improves retrieval accuracy
2. Embedding Generation & Vector Storage
To enable efficient search and retrieval, the system converts document chunks into vector embeddings using Sentence Transformers (all-MiniLM-l6-v2). These embeddings are then stored in a FAISS (Facebook AI Similarity Search) vector database, allowing fast retrieval of relevant document sections when a query is received.

Embedding Model: sentence-transformers/all-MiniLM-l6-v2
Vector Store: FAISS (used for fast similarity searches)
Why FAISS?
Optimized for fast nearest neighbor search
Scales well for large datasets
Provides efficient semantic search capabilities
3. Local LLM for Response Generation
Once the system retrieves relevant document chunks, it uses a local Mistral-7B model to generate answers. The model is loaded using CTransformers, allowing it to run efficiently on both CPU and GPU.

LLM Model: mistral-7b-v0.1.Q3_K_L.gguf
Model Configuration:
max_new_tokens: 1024 (Generates detailed responses)
temperature: 0.1 (Ensures deterministic responses)
top_k: 50 (Controls token selection randomness)
top_p: 0.9 (Adjusts probability distribution)
Device Adaptability:
Automatically detects and utilizes GPU (CUDA) if available
Falls back to CPU execution when GPU is not available
The LLM processes user questions by integrating retrieved document context and generating a human-like response.

4. API Endpoint using FastAPI
To make the system accessible, a FastAPI-based backend is implemented. The /ask endpoint allows users to submit queries and receive answers in JSON format.

5. Gradio Web Interface
For a user-friendly experience, the project includes a Gradio-based UI, allowing users to interact with the system without using the API directly.

Features of Gradio Interface:
Simple text input for user queries
Real-time responses generated using the RAG pipeline
Web-based UI that can be accessed locally
To launch the Gradio UI, the script runs both FastAPI (on port 8000) and the Gradio app concurrently using Python’s threading module.
