from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
import torch


loader = PyPDFLoader("ash/1706.03762v7.pdf")  
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


db = FAISS.from_documents(chunks, embeddings)


local_llm = "model2/mistral-7b-v0.1.Q3_K_L.gguf"  

config = {
    'max_new_tokens': 1024,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

llm = CTransformers(
    model=local_llm,
    model_type="mistral",
    lib="cuda" if device.type == "cuda" else "avx2",  
    **config
)

print("LLM Initialized...")


from langchain.prompts import ChatPromptTemplate

template = """
Question: {question}
Instruction: You are an assistant for students. Answer the question based on the content in the PDF file.
Provide the page number and reference for the given topic if available.
Answer: {answer}
"""

prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"question": RunnablePassthrough(), "answer": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


app = FastAPI()

class QueryModel(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: QueryModel):
    try:
        result = rag_chain.invoke(query.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def ask_question_via_api(question):
    try:
        response = requests.post("http://localhost:8000/ask", json={"question": question})
        if response.status_code == 200:
            return response.json().get("answer", "No answer found.")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

iface = gr.Interface(
    fn=ask_question_via_api,
    inputs="text",
    outputs="text",
    title="RAG System Interface",
    description="Ask questions about the PDF document and get answers powered by the RAG system."
)


if __name__ == "__main__":
    import threading
    import uvicorn

    
    fastapi_thread = threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": 8000})
    fastapi_thread.start()

    
    iface.launch()