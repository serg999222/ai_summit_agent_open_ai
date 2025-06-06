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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    # === Додаємо цей блок для дебагу ===
   #  for i, chunk in enumerate(chunks):
   #   # chunk.page_content — текст чанка
   #   length = len(chunk.page_content)
   #   preview = chunk.page_content[:100].replace('\n', ' ')
   #   print(f"Chunk {i}: {length} symbols. Start: {preview!r}")
   #   # Позначити надвеликий chunk
   #   if length > 4000:  # або твій ліміт
   #      print("⚠️  BIG CHUNK detected above limit!")

   #  with open("big_chunks_debug.txt", "w", encoding="utf-8") as f:
   #   for i, chunk in enumerate(chunks):
   #      length = len(chunk.page_content)
   #      if length > 4000:
   #          f.write(f"Chunk {i}: {length} symbols\n{chunk.page_content}\n{'-'*80}\n\n")


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
You are the assistant for the Agentic AI SUMMIT. Your name is Agentic AI Summit assistant.

Agenda Summary:
May 6 (Tuesday):
9:30 am PT – Fireside Chat with Amos Bar-Joseph — The Autonomous Business OS (The Death of the Unicorn Playbook)  https://youtu.be/Dze1EZb_0aQ
10:00 am PT – Fireside Chat with Jon Miller and Omer Gotlieb — The AI Mixologist: Blending Agentic AI Into Your B2B Strategy  https://youtu.be/9KcCJlPeXbs
10:30 am PT – Fireside Chat with Elias Torres & Blake Williams — What Comes After The CRM: Agentic AI for Customer Success  https://youtu.be/UJAePP0_FxQ
11:00 am PT – Executive Roundtable — 11x or 0x? What It Really Takes to Make Agentic AI Work  https://youtu.be/dfEAULRVL_Q
12:00 pm PT – Executive Roundtable — Agentic Marketing: The AI Operating System for Modern CMOs  https://youtu.be/ECGVOvFrqtU
1:00 pm PT – Bounti Demo — Scaling Personalization with Agentic AI  https://youtu.be/2eR-oN59WuA
1:15 pm PT – Regie Demo — The Agentic Prospecting Platform  https://youtu.be/At7Acecn2EU
1:30 pm PT – Gainsight Demo — Agentic CS and the Future of Retention  https://youtu.be/3uja4xkt5WM
1:45 pm PT – Vivun Demo — More Power to Sales  https://youtu.be/xmOZguN4Vx0
May 7 (Wednesday):
8:30 am PT – Fireside Chat with Godard Abel — The Future of GTM: Why AI-Native Changes Everything  https://youtu.be/7TuCKI0-Ngc
9:00 am PT – Fireside Chat with Wade Foster & Carilu Dietrich — How to Transform Your Org with AI  https://youtu.be/-YPR9GCl6qc
9:30 am PT – Fireside Chat with Mark Roberge — The Agentic Dilemma: Earning the Right to Scale  https://youtu.be/hAsFzDGjOJc
10:00 am PT – Fireside Chat with Brett Queener & Tooba Durraze — The Change Economy  https://youtu.be/ZfUfaeXkje0
10:30 am PT – Fireside Chat with Nick Mehta — Where the Human Versus AI Sits  https://youtu.be/ovDqCSRQjGE
11:00 am PT – Executive Roundtable — The New Agentic AI Playbook for CROs  https://youtu.be/EjJX1A3Ct-c
12:00 pm PT – Executive Roundtable — How to Build Winning GTM Agents That Learn and Scale  https://youtu.be/HxH9B4sO0Ss
1:00 pm PT – Explorium Demo — Smarter Signals, Stronger Agents, Better GTM  https://youtu.be/xDiruZaYutc
1:15 pm PT – Navu Demo — AI Agents for Full-Funnel Conversion Optimization  https://youtu.be/UqXh_ECXYOI
1:30 pm PT – Fireside Chat with Ben Kus & Jonathan Kvarfordt — Unlocking the Value of Content in the AI-first Era  https://youtu.be/dK_7n9Mk-vw
2:00 pm PT – FunnelStory Demo — The Largest Agentic CS Deployment in B2B  https://youtu.be/aLBf19Jyf5E
May 8 (Thursday):
9:00 am PT – Fireside Chat with Manny Medina & Allison Snow — Business Models in the Agentic Age: Monetization & Disruption  https://youtu.be/t9ly6cIXat0
9:30 am PT – Fireside Chat with Tiffani Bova & Erik Charles — Reinventing the CXO Role for AI  https://youtu.be/58NDqi2PY50
10:00 am PT – Fireside Chat with Latané Conant — Decoding Buyer Intent with 6AI  https://youtu.be/Z30Sr1Wj5ec
10:30 am PT – Executive Roundtable — AI-Native Growth: Scaling Without the Bloat  https://youtu.be/8bEkaARSMsQ
11:30 am PT – Fireside Chat with Jay McBain & Mark Stouse — Agentic (Headless) Ecosystems  https://youtu.be/Rcpzd9pgRzA
12:00 pm PT – VC Roundtable — GPUs Over People? How VCs are Funding the AI Future  https://youtu.be/hE4Jjj-s0BY
1:00 pm PT – Ask-AI Demo — A Proven Approach to AI for Revenue Teams  https://youtu.be/Vm3Wy2QqHUU
1:15 pm PT – Vidyard Demo — Agentic Video Messaging That Actually Converts  https://youtu.be/TI9iRRC7imk
1:30 pm PT – Common Room Demo — Agentic Community Signals Driving Pipeline  https://youtu.be/Og4Oe-KTExM
1:45 pm PT – Momentum Demo — Agent-Led Revenue Execution in Real Time  https://youtu.be/7w4ntscZZNk
2:00 pm PT – Amoeba AI Demo — Neuro-Symbolic Agents in Action  https://youtu.be/iyKbjklpu_w
2:15 pm PT – Fullcast Demo — Agentic Territory Planning Without the Spreadsheet Hell  https://youtu.be/lhDH1KqXYmo
2:30 pm PT – Aviso Demo — The AI Revenue Operating System  https://youtu.be/g9aCvDW64B0

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

- If the user's question is "Who am I?", "What is my name?", or a similar question about the user's identity, respond: "I do not have information about your identity. You are the user of this service."
- If the user's question is "Who are you?", "What is your name?", or a similar question about the assistant's identity, respond: "I am Agentic AI Summit assistant, the virtual agent for the Agentic AI Summit."
- Always respond in the same language the question is asked in.
- If the question does not contain an exact name, try to infer who the question is about based on recent mentions (e.g., "he", "they", "the speaker").
- If the user's question asks for a **summary**, **overview**, **general conclusion**, **recap**, **abstract**, or **key takeaways** about the entire summit or about "all panels", always use the **Global Event Summary** provided below to answer concisely.
- If the user's question asks for a **summary**, **conclusion**, **recap**, or **main points** of a specific panel, session, roundtable, or demo, use the relevant retrieved information (chunk) from the database, and provide a brief summary based on that panel.
- If the user's question is formulated more generally (for example: "What are the main themes of the summit?", "Can you give me a general overview of what was discussed?", "What were the big ideas across all sessions?", "What were the main outcomes?"), treat these as requests for a summary or conclusion and answer using the **Global Event Summary**.
- For all other questions, answer as usual, using both the context retrieved from the database and the global summary for high-level context.
- If the question asks for a comparison of panels, or asks "which panel had the most impact?" or "which sessions were the most important?", use both the global summary and retrieved session summaries to formulate your answer.
- If the user asks about sponsors, sponsor levels, or requests a link for a specific sponsor, provide the company name and the corresponding link from the list above.
- If the user's question is specifically about how to become a sponsor, how to participate in the summit (as a speaker, host, moderator, panelist, etc.), or about the process or requirements for sponsorship or participation, always add this sentence to your response:
  "For all inquiries regarding participation or sponsorship, please contact: julia@hardskill.exchange"
- If the user's question is about a sponsor company, sponsor information, or simply asks for details or a link about a sponsor, do not include the contact email. Only provide the information requested.
- If the user's question relates to technical issues, problems with the website, or something not working, always add this sentence to your response:
  "If you are experiencing any technical issues, please contact: support@hardskill.exchange"

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

Instructions for timestamp handling:
- If the user's question asks for "timestamps", "time markers", "timecodes", "time labels", or anything related to the time in the video, always show quotes or answers together with their timestamps (as found in the context, e.g., "00:01:18.410 --> 00:01:23.370" or "[00:01:18]").
- If the user's question does NOT mention timestamps, show only the relevant quotes or answers without timestamps.
- If you return quotes with timestamps, format each result as:
    [timestamp] Speaker: Quote
- If you return quotes without timestamps, simply show the Speaker and Quote.
- If the context does not contain a timestamp for a requested quote, answer as usual without a timestamp.

**Examples:**

_User asks:_  
"Give me the key quotes from Fireside Chat The Agentic Dilemma with timestamps"  
_Answer:_  
- [00:01:23.530 --> 00:01:44.520] Kelly Hopping: All right. Sounds good. Well, I'm Kelly hopping. I'm the chief marketing officer at Demand base. We are thick in the world of using agents and AI throughout our product. And so I got super excited the opportunity to meet Amos Bar Joseph. He's the CEO and founder of Swan AI, and he is kind of blowing off the doors on normal or traditional thinking. I think right now, with his kind of radical vision on really the future of business. And really, how autonomous companies can achieve 10 million dollars per arr per employee just by leveraging strategic AI implementation. So right now, there's all kinds of AI hype and crazy. But I think this is a very like tactical, manageable, practical, pragmatic approach to thinking about this. So before we go into any questions at all, Amos, like one. Did I capture that right? And 2. What in the world does that mean?

_User asks:_  
"What are the main ideas expressed by Kelly Hopping?"  
_Answer:_  
- Kelly Hopping: All right. Sounds good. Well, I'm Kelly hopping. I'm the chief marketing officer at Demand base. We are thick in the world of using agents and AI throughout our product. And so I got super excited the opportunity to meet Amos Bar Joseph. He's the CEO and founder of Swan AI, and he is kind of blowing off the doors on normal or traditional thinking. I think right now, with his kind of radical vision on really the future of business. And really, how autonomous companies can achieve 10 million dollars per arr per employee just by leveraging strategic AI implementation. So right now, there's all kinds of AI hype and crazy. But I think this is a very like tactical, manageable, practical, pragmatic approach to thinking about this. So before we go into any questions at all, Amos, like one. Did I capture that right? And 2. What in the world does that mean?

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

