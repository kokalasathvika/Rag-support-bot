# test_setup.py — run this to confirm everything works
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
response = llm.invoke("Say: setup successful")
print(response.content)