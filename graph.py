import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

embedding = HuggingFaceEmbeddings(
 model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
 persist_directory="chroma_db",
 embedding_function=embedding
)

retriever = db.as_retriever(search_kwargs={"k":3})

class State(TypedDict):
    question:str
    answer:str

def process_query(state):
    question = state["question"]

    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are customer support assistant.
Answer only from context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}

def route_query(state):
    q = state["question"].lower()

    urgent_words = ["payment failed","money deducted","angry","complaint","human","manager"]

    if any(word in q for word in urgent_words):
        return "escalate"

    return "normal"

def escalate(state):
    return {"answer":"Your issue has been escalated to human support."}

builder = StateGraph(State)

builder.add_node("process", process_query)
builder.add_node("escalate", escalate)

builder.set_conditional_entry_point(
    route_query,
    {
        "normal":"process",
        "escalate":"escalate"
    }
)

builder.add_edge("process", END)
builder.add_edge("escalate", END)

graph = builder.compile()