import streamlit as st
from streamlit_chat import message
from utils import *
from dotenv import load_dotenv
import variables
import os
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()
st.session_state['HuggingFace_API_Key'] = os.environ.get("HuggingFace_API_Key")
st.session_state['PINECONE_API_KEY'] = os.environ.get("PINECONE_API_KEY")
st.session_state['Pinecone_API_Key'] = os.environ.get("PINECONE_API_KEY")

# Creating Session State Variable
if 'HuggingFace_API_Key' not in st.session_state:
    st.session_state['HuggingFace_API_Key'] = ''
if 'Pinecone_API_Key' not in st.session_state:
    st.session_state['Pinecone_API_Key'] = ''
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

#
st.title('Financial Mathematics + Machine Learning Chatbot🤖') 
# load_button = st.sidebar.button("Load data to Pinecone", key="load_button")
# ---------------------------------------------------------- Chat without container ---------------------------------------------------------- #
# Captures User Inputs
to_ask = st.text_input('Ask away!', key="prompt")  # The box for the text prompt
# document_count = st.slider('No.Of links to return 🔗 - (0 LOW || 5 HIGH)', 0, 5, 2,step=1)
submit = st.button("Query")

# Base system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know."
    "\n\n"
    "{context}"
)

# ---------------------------------------------------------- Without memory ---------------------------------------------------------- #
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# ---------------------------------------------------------- With memory ---------------------------------------------------------- #

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise response as a simple QA chatbot would do."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Variable Initialization
chain = None

if submit:
    # print(st.session_state) # For debugging only

    #Proceed only if API keys are provided
    if st.session_state['HuggingFace_API_Key'] !="" and st.session_state['PINECONE_API_KEY']!="" :

        # Create a RAG chain
        if chain is None:            

            # ----------------------------------------- Without memory ----------------------------------------- #
            # question_answer_chain = create_stuff_documents_chain(variables.LLM, prompt)
            # chain = create_retrieval_chain(variables.INDEX.as_retriever(search_kwargs={'k': 6}), question_answer_chain)

            # ----------------------------------------- With memory ----------------------------------------- #
            history_aware_retriever = create_history_aware_retriever(
                        variables.LLM, variables.INDEX.as_retriever(search_kwargs={'k': 6}), contextualize_q_prompt)
            question_answer_chain = create_stuff_documents_chain(variables.LLM, qa_prompt)
            chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        response = chain.invoke({"input": to_ask, "chat_history": st.session_state['chat_history']})
        st.session_state['chat_history'].extend(
            [
                HumanMessage(content=to_ask),
                AIMessage(content=response["answer"]),
            ]
        )

        # print(to_ask) # For debugging only
        # print(response) # For debugging only
        st.write(response['answer'])
    
    else:
        st.sidebar.error("Please provide API keys.....")