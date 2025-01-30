from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import ChatOpenAI
from langserve import add_routes

from langchain_openai import AzureChatOpenAI
from langchain.llms import Ollama

import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
os.environ["OPENAI_API_TYPE"] = os.getenv("OPENAI_API_TYPE")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A Simple API Server",
)

# add_routes(
#     app,
#     AzureChatOpenAI(),
#     path="/azure_openai",
# )

# Azure llm model
model = AzureChatOpenAI(model="gpt-4o-mini", temperature=0.7)
prompt1 = ChatPromptTemplate.from_template("Write an essay about {topic} in 100 words.")

add_routes(
    app,
    prompt1|model,
    path="/essay",
)

# Ollama llm model
llm = Ollama(model="mistral")
prompt2 = ChatPromptTemplate.from_template("Generate a poem on {topic} for children.")

add_routes(
    app,
    prompt2|llm,
    path="/poem",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)