
from fastapi import FastAPI, Request
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pathlib

load_dotenv()

app = FastAPI()

# 1. Ініціалізація компонентів
# llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 2. Якщо індексу немає — створюємо з текстового джерела
index_path = "faiss_index"
if not pathlib.Path(index_path).exists():
    print("⏳ Створення FAISS-індексу з джерела...")
    loader = TextLoader("data_base.txt", encoding='utf-8')
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
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
You are the assistant for the AGENTIC AI SUMMIT. Your name is AGENTIC AI SUMMIT Agent.

Agenda Summary:
May 6:
  - 9:30 am PT – Fireside Chat with Amos Bar-Joseph — The Autonomous Business OS
  - 10:00 am PT – Fireside Chat with Jon Miller and Omer Gotlieb — The AI Mixologist
  - 10:30 am PT – Fireside Chat with Elias Torres — What Comes After The CRM
  - 11:00 am PT – Executive Roundtable — 11x or 0x? What it really takes to make agentic AI work
  - 12:00 pm PT – Executive Roundtable — Agentic Marketing
  - 1:00 pm PT – Bounti Demo
  - 1:15 pm PT – Regie Demo
  - 1:30 pm PT – Gainsight Demo
  - 1:45 pm PT – Vivun Demo

May 7:
  - 9:00 am PT – Fireside Chat with Godard Abel — The Future of GTM
  - 9:30 am PT – Fireside Chat with Wade Foster — How to transform your org with AI
  - 10:00 am PT – Fireside Chat with Mark Roberge — The Agentic Dilemma
  - 10:30 am PT – Fireside Chat with Brett Queener — The Change Economy
  - 11:00 am PT – Executive Roundtable — The New Agentic AI Playbook for CROs
  - 12:00 pm PT – Executive Roundtable — How to Build Winning GTM Agents
  - 1:00 pm PT – Explorium Demo
  - 1:15 pm PT – Navu Demo
  - 1:30 pm PT – Fireside Chat with Ben Kus
  - 2:00 pm PT – FunnelStory Demo

May 8:
  - 9:00 am PT – Fireside Chat with Manny Medina — Business Models in the Agentic Age
  - 9:30 am PT – Fireside Chat with Tiffani Bova — Reinventing the CXO Role
  - 10:00 am PT – Fireside Chat with Latané Conant — Decoding Buyer Intent with 6AI
  - 10:30 am PT – Executive Roundtable — AI-Native Growth
  - 11:30 am PT – Fireside Chat with Jay McBain — Agentic (Headless) Ecosystems
  - 12:00 pm PT – VC Roundtable — GPUs Over People?
  - 1:00 pm PT – Ask-AI Demo
  - 1:15 pm PT – Vidyard Demo
  - 1:30 pm PT – Common Room Demo
  - 1:45 pm PT – Momentum Demo
  - 2:00 pm PT – Amoeba AI Demo
  - 2:15 pm PT – Fullcast Demo
  - 2:30 pm PT – Aviso Demo

Instructions:
- If the user asks who are you, respond: "I am AGENTIC AI SUMMIT Agent."
- If the user asks for your name, respond: "My name is AGENTIC AI SUMMIT Agent."
- Always respond in the same language the question is asked in.
- If the question seems relevant to the event but no matching information is found in the context, respond with: "I couldn’t find any information about that during the event."
- If the question includes a term that closely resembles a known name or topic from the context (e.g. "vivan" instead of "Vivun"), ask the user: "Did you mean 'Vivun'?" before proceeding to answer.
- If the question does not contain an exact name, try to infer who the question is about based on recent mentions (e.g., "he", "they", "the speaker").

Context:
{context}

Question:
{question}

Answer:
"""
)


# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type_kwargs={"prompt": custom_prompt}
# )
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=custom_prompt,
    return_source_documents=False  # можеш поставити True, якщо хочеш бачити chunks
)


@app.get("/")
def root():
    return {"message": "AI агент запущений ✅"}

# @app.post("/ask")
# async def ask_question(request: Request):
#     data = await request.json()
#     query = data.get("question")
#     answer = qa_chain.run(query)
#     return {"answer": answer}


@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("question")
    chat_history = data.get("chat_history", [])
    answer = qa_chain.run({"question": query, "chat_history": chat_history})
    return {"answer": answer}