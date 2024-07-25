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

# ---------------------------------------------------------- Chat with container ---------------------------------------------------------- #
response_container = st.container()
# Here we will have a container for user input text box
container = st.container()
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Ask away!:", key='input', height=50)
        submit = st.form_submit_button(label='Query')
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
            if st.session_state['HuggingFace_API_Key'] !="" and st.session_state['PINECONE_API_KEY']!="" :
                if chain is None:
                    # ----------------------------------------- Without memory ----------------------------------------- #
                    # question_answer_chain = create_stuff_documents_chain(variables.LLM, prompt)
                    # chain = create_retrieval_chain(variables.INDEX.as_retriever(search_kwargs={'k': 6}), question_answer_chain)

                    # ----------------------------------------- With memory ----------------------------------------- #
                    history_aware_retriever = create_history_aware_retriever(
                                variables.LLM, variables.INDEX.as_retriever(search_kwargs={'k': 6}), contextualize_q_prompt)
                    question_answer_chain = create_stuff_documents_chain(variables.LLM, qa_prompt)
                    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                st.session_state['messages'].append(user_input)
                response = chain.invoke({"input": user_input, "chat_history": st.session_state['chat_history']})
                st.session_state['chat_history'].extend(
                    [
                        HumanMessage(content=user_input),
                        AIMessage(content=response["answer"]),
                    ]
                )
                st.session_state['messages'].append(response["answer"])
            

                with response_container:
                    for i in range(len(st.session_state['messages'])):
                            if (i % 2) == 0:
                                message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
                            else:
                                message(st.session_state['messages'][i], key=str(i) + '_AI')
            else:
                st.sidebar.error("Please provide API keys.....")