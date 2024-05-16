import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv

load_dotenv("./.env")
groq_api_key = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GEMINI_API_KEY
    )
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_spilitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_document = st.session_state.text_spilitter.split_documents(
        st.session_state.docs[:50]
    )
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_document, st.session_state.embeddings
    )


st.title("CHATGROQ Demo")
llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")

prompt = ChatPromptTemplate.from_template(
    """
        Answer the question based on the provided conetext only.
        Please provide the most accurate response based on the question 
        <context>
        {context}
        </context>
        Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input you prompt here")

if prompt:
    start_time = time.process_time()
    response = retriever_chain.invoke({"input": prompt})
    print("Response time: ", time.process_time() - start_time)
    st.write(response["answer"])

    # With a steamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------")
