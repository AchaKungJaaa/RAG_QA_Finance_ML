import torch
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone 
from pinecone import Pinecone as PineconeClient
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
# Function to pull index data from Pinecone

# Function to create embeddings instance
def create_embeddings():
    """
        Create a text embedder that will be used through out the process

        Input (s): None

        Return:
            An embedding object that can be used to project text to a vector space
    """

    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=os.environ.get("EMBEDDER_NAME"),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs)

    return embeddings

def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):

    """
        Load an existing index from Pinecone

        Inputs (s):
            pinecone_apikey (str): pinecon API key
            pinecone_environment (str): Name of the cloud service used to host the pinecone index
            pinecone_index_name (str): The name of an index to be used
            embeddings: an embedding model, used to embed the query to the same space as the vector in the index
        Return:
            A Pinecone object ready to perform similarity search on the document
    """

    PineconeClient(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name
    #PineconeStore is an alias name of Pinecone class, please look at the imports section at the top :)
    index = Pinecone.from_existing_index(index_name, embeddings)
    return index


# This function will help us in fetching the top relevent documents from our vector store - Pinecone
def get_similar_docs(index, query, k=5):
    """
        Perform a similarity search on the index based on the given query

        Input (s):
            index: A Pinecone object. Should have the similarity_search method defined. Can be obtained from pull_from_pinecone() method
            query (str): A user query 
            k (int): the number of documents to retreive. Default to 5
        
        Return:
            k most similar documents based on the query provided 
    """
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

#This function will help us get the answer to the question that we raise
def get_answer(chain, index, query):
  """
    Run the query on the chain and index provided

    Input (s):
        chain: A chained processed provided by langchain
        index: A Pinecone object. Should have the similarity_search method defined. Can be obtained from pull_from_pinecone() method
        query (str): A user query 

    Return:
        A response from an LLM
  """
  relevant_docs = get_similar_docs(index, query)
  response = chain.run(input_documents=relevant_docs, question=query)
  return response