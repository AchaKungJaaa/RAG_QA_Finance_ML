Hi! Welcome to my first presonal project: QA-RAG on Textbooks.

Motivation:
My interest span to many fields of knowledge, Finance, Economics, Statistics, Machine Learning and Deep Learning, Linear Control System, Advanced mathematics (Complex Analysis, Optimization techniques, Laplace transform ...) etc.
Due to my interest..., My collection of textbooks becomes increasingly large and I can't keep up with them! So, I create this QA-Rag system for myself to get a quick answer to some of my questions without haveing to skim through all of them!

Setup:
I mainly use Langchain, Pinecone Database and GroqAPI for the main functionality of this project. The Streamlit library is used for the demo on website.
In this project, I only use two PDF files for the demo, ISLR textbook and Financial Mathematics textbook. 

Functionality:
- This application can act as both conversational bot and RAG bot, able to accurately query the content related to the user's query..
- This application can recall past conversation through the chat history.

Limitaions:
- Due to being a free LLM version, the context lenght and the token number of this application is quite limited.

Resource:
I would like to shout out to Sharath Raju. I have enrolled in his Langchain course on Udemy and that course has provided me with great source codes and gave me an idea of how to get started so this project is possible!
Other that Sharath Raju's course, I use the following site for tutorial and code template.

Conversational Rag: https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/
