from fastapi import FastAPI, Request
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
import os
import pathlib

load_dotenv()

app = FastAPI()

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

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
    input_variables=["context", "question", "chat_history"],
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

Summit Sponsors List
Below is the list of sponsors for the summit. If the user asks for the sponsors, their levels, or their official links, use this information.

Priority Sponsors
Gainsight — https://www.gainsight.com/
Momentum — https://www.momentum.io/
Bounti — https://www.bounti.ai/
Vivun — https://vivun.com/
Regie — https://www.regie.ai/
Ask-AI — https://www.ask-ai.com/
Explorium — http://explorium.ai/
Navu — https://navu.co/
Amoeba AI — https://www.amoeb.ai/
Aviso — https://www.aviso.com/

Standard Sponsors
Fullcast — https://www.fullcast.com/
Nooks — https://www.nooks.ai/
Alta — https://www.altahq.com/
Vidyard — https://www.vidyard.com/
Common Room — https://www.commonroom.io/
FunnelStory — https://funnelstory.ai/
TheySaid — https://www.theysaid.io/
SiftHub — http://sifthub.io/
SalesPeak — https://salespeak.ai/
Averi — https://www.averi.ai/
Affinity — https://www.affinity.co/
Substrata — https://www.substrata.me/
Cloudlead — https://www.cloudlead.co/



Instructions:
- If the user asks who are you, respond: "I am AGENTIC AI SUMMIT Agent."
- If the user asks for your name, respond: "My name is AGENTIC AI SUMMIT Agent."
- Always respond in the same language the question is asked in.
- If the question seems relevant to the event but no matching information is found in the context, respond with: "I couldn’t find any information about that during the event."
- If the question includes a term that closely resembles a known name or topic from the context (e.g. "vivan" instead of "Vivun"), ask the user: "Did you mean 'Vivun'?" before proceeding to answer.
- If the question does not contain an exact name, try to infer who the question is about based on recent mentions (e.g., "he", "they", "the speaker").
- If the user's question asks for a **summary**, **overview**, **general conclusion**, **recap**, **abstract**, or **key takeaways** about the entire summit or about "all panels", always use the **Global Event Summary** provided below to answer concisely.
- If the user's question asks for a **summary**, **conclusion**, **recap**, or **main points** of a specific panel, session, roundtable, or demo, use the relevant retrieved information (chunk) from the database, and provide a brief summary based on that panel.
- If the user's question is formulated more generally (for example: "What are the main themes of the summit?", "Can you give me a general overview of what was discussed?", "What were the big ideas across all sessions?", "What were the main outcomes?"), treat these as requests for a summary or conclusion and answer using the **Global Event Summary**.
- For all other questions, answer as usual, using both the context retrieved from the database and the global summary for high-level context.
- If the question asks for a comparison of panels, or asks "which panel had the most impact?" or "which sessions were the most important?", use both the global summary and retrieved session summaries to formulate your answer.
- If the user asks about sponsors, sponsor levels, or requests a link for a specific sponsor, provide the company name and the corresponding link from the list above.

Examples of requests that should trigger use of the Global Event Summary:
- "Can you summarize the summit?"
- "What was the overall conclusion of the Agentic AI Summit?"
- "Give me a recap of all the panels."
- "What were the main takeaways from all sessions?"
- "Can you provide an overview of the summit?"
- "What did the summit focus on?"
- "What were the central themes discussed during the summit?"
- "Can I get a general abstract or conclusion about the event?"

If the user requests a **summary of a specific panel**, only use the relevant panel summary from the database.  
If the request is about the **summit as a whole**, always use the Global Event Summary below.

Global Event Summary:
The Agentic AI Summit united top thinkers, founders, and innovators to explore how agentic AI is fundamentally reshaping business. Each session provided unique perspectives on leveraging autonomous agents for growth, efficiency, and better customer experiences. The central theme was the shift from rule-based automation to reasoning, collaboration, and augmentation—unlocking the next wave of business transformation.


Context:
{context}

Chat history:
{chat_history}

Question:
{question}

Answer:
"""
)

# Конструюємо сам chain:
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
)

@app.get("/")
def root():
    return {"message": "AI агент запущений ✅"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    chat_history = data.get("chat_history", [])
    
    # Примусово переводимо у list of tuples
    chat_history_tuples = [tuple(item) for item in chat_history]

    print("question:", repr(question), type(question))
    print("chat_history:", repr(chat_history_tuples), type(chat_history_tuples))
    
    result = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history_tuples
    })
    return {"answer": result}

