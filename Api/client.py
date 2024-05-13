import requests
import streamlit as st


def get_essay_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke", json={"input": {"topic": input_text}}
    )
    return response.json()["output"]["content"]

def get_poem_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke", json={"input": {"topic": input_text}}
    )
    return response.json()["output"]["content"]

st.title('LangChain Demo using API')
input_text1 = st.text_input("Write an essay on...")
input_text2 = st.text_input("Write a poem on...")

if input_text1:
    st.subheader("Essay: ")
    st.write(get_essay_response(input_text1))

if input_text2:
    st.subheader("Poem: ")
    st.write(get_poem_response(input_text2))