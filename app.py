# from fastapi import FastAPI, Request
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# import os

# load_dotenv()

# app = FastAPI()

# # Ініціалізація компонентів
# llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
# embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
# vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# retriever = vectorstore.as_retriever()
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# @app.get("/")
# def root():
#     return {"message": "AI агент запущений ✅"}

# @app.post("/ask")
# async def ask_question(request: Request):
#     data = await request.json()
#     query = data.get("question")
#     answer = qa_chain.run(query)
#     return {"answer": answer}


from fastapi import FastAPI, Request
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pathlib

load_dotenv()

app = FastAPI()

# 1. Ініціалізація компонентів
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 2. Якщо індексу немає — створюємо з текстового джерела
index_path = "faiss_index"
if not pathlib.Path(index_path).exists():
    print("⏳ Створення FAISS-індексу з джерела...")
    loader = TextLoader("data_base.txt", encoding='utf-8')
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print("✅ Індекс створено.")
else:
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print("✅ Індекс завантажено.")

retriever = vectorstore.as_retriever()

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the assistant for the AGENTIC AI SUMMIT. Your knowledge is strictly limited to the information provided in the context retrieved from the summit's knowledge base.

Instructions:
- If the question is directly or indirectly related to the AGENTIC AI SUMMIT (topics, discussions, speakers, sessions, etc.) but the context does not contain enough information to answer it, respond by saying that the question is related to the event, but you currently don't have information about it.
- If the question is unrelated to the event, respond with: "I can only answer questions related to this event."

Context:
{context}

Question:
{question}

Answer:
"""
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)



@app.get("/")
def root():
    return {"message": "AI агент запущений ✅"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("question")
    answer = qa_chain.run(query)
    return {"answer": answer}
