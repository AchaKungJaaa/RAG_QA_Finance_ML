
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from utils import *
import os
from langchain_pinecone.vectorstores import PineconeVectorStore

load_dotenv()

LLM = ChatGroq(model="llama3-8b-8192")
EMBEDDINGS = create_embeddings()
INDEX = PineconeVectorStore(
                            index_name=os.environ.get("PINECONE_INDEX"),
                            embedding=EMBEDDINGS)