# from langchain_openai import AzureChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# import streamlit as st
# import os 

# from dotenv import load_dotenv

# #Environment variables call
# load_dotenv()

# # print(os.environ.get("AZURE_OPENAI_ENDPOINT"))

# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
# os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
# os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
# os.environ["OPENAI_API_TYPE"] = os.getenv("OPENAI_API_TYPE")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

# # Creating a new message if there isn't one
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         # ("system", "You're a helpful assistant. Please provide responses to the queries."),
#         SystemMessage(content="You're a helpful assistant. Please provide responses to the queries.", additional_kwargs={"role":'system'})
#     ]

# # Creating chatbot
# prompt = ChatPromptTemplate.from_messages(st.session_state.messages)

# # Streamlit framework

# st.title("ChatBot using LangChain and AzureOpenAI")

# # Input box
# input_text = st.text_input("Enter any topic you want.")

# # AzureOpenAI LLM call
# llm = AzureChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# output_parser = StrOutputParser()

# # Chain
# chain = prompt | llm | output_parser

# # Print messages
# for message in st.session_state.messages:
#     if 'role' in message.additional_kwargs:
#         with st.chat_message(message.additional_kwargs["role"]):
#             print(message)
#             st.markdown(message.content) 

# # Invoking the model
# if input_text: 
#     st.chat_message("user").markdown(input_text)
    
#     # st.session_state.messages.append("user", f"Query: {input_text}")
#     # st.write(chain.invoke({'question': input_text}))

#     st.session_state.messages.append(HumanMessage(content=input_text, additional_kwargs={"role": 'human'}))
#     response=chain.invoke({'question': input_text})
#     with st.chat_message("assistant"):
#         st.markdown(response)
    
#     st.session_state.messages.append(AIMessage(content=response, additional_kwargs={"role": 'ai'}))


from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import streamlit as st
import os 

from dotenv import load_dotenv

#Environment variables call
load_dotenv()

# print(os.environ.get("AZURE_OPENAI_ENDPOINT"))

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
os.environ["OPENAI_API_TYPE"] = os.getenv("OPENAI_API_TYPE")

# Creating chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a helpful assistant and your name is Siri. Please provide responses to the queries."),
        ("user", "Query: {question}")
    ]
)

# Streamlit framework

st.title("ChatBot using LangChain and AzureOpenAI")

# Input box
input_text = st.text_input("Enter any topic you want.")

# AzureOpenAI LLM call                  
llm = AzureChatOpenAI(model="gpt-4o-mini", temperature=0.7)
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

# Invoking the model
if input_text: 
    st.write(chain.invoke({'question': input_text}))