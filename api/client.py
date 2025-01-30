import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_azureopenai_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke",
                             json={'input':{'topic':input_text}})
    return response.json()['output']['content']

def get_mistril_response(input_text2):
    response=requests.post("http://localhost:8000/poem/invoke",
                             json={'input':{'topic':input_text2}})
    return response.json()['output']
    
# Streamlit framework
st.title("Langchain with AzureOpenAI and Mistril")
input_text = st.text_input("Write an essay on...")
input_text2 = st.text_input("Write a poem on...")

if input_text:
    st.write(get_azureopenai_response(input_text))

if input_text2:
    st.write(get_mistril_response(input_text2))