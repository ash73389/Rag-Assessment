# Rag-Assessment
This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about a PDF document and receive relevant answers. It is built using FastAPI, LangChain, FAISS, and a local Mistral-7B model to enable efficient information retrieval and response generation.

The system first loads a PDF document using PyPDFLoader from LangChain. Since large documents are difficult to process as a whole, the text is split into smaller chunks using RecursiveCharacterTextSplitter, ensuring that relevant information can be retrieved effectively.

To facilitate fast and accurate search, the project uses FAISS (Facebook AI Similarity Search) to store and index vector embeddings of the document. The embeddings are generated using Sentence Transformers (all-MiniLM-l6-v2), making it possible to retrieve relevant sections of the document based on user queries.

For response generation, the system utilizes a Mistral-7B model loaded with CTransformers. This model is configured to run on either CPU or GPU, ensuring flexibility in execution. The retrieved document chunks are fed into the model along with the user’s query, and the LLM generates a relevant response.

A FastAPI backend exposes an API endpoint (/ask), where users can send a question, and the system will return an AI-generated response based on the document content. Additionally, a Gradio interface provides a simple web-based UI for users to interact with the system without needing API calls.

This project demonstrates how FastAPI, LangChain, FAISS, and LLMs can be integrated to create a powerful document-based Q&A system. It is useful for research, education, and enterprise applications where retrieving information from large documents is necessary.

