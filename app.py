from fastapi import FastAPI, Request
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Ініціалізація компонентів
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.load_local("faiss_index", embeddings)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.get("/")
def root():
    return {"message": "AI агент запущений ✅"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("question")
    answer = qa_chain.run(query)
    return {"answer": answer}
