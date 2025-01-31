from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## Load the document to be queried.
 
loader=PyPDFLoader('Aneeshkrishna_Resume.pdf')

docs=loader.load()
# docs

## Split the text into chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

chunk_docs=text_splitter.split_documents(docs)
# chunk_docs

## Vector Embeddings and Vector stores
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

db=FAISS.from_documents(chunk_docs,AzureOpenAIEmbeddings(model='gpt-text-embedding-3-large'))
db

## Query the vector store for similarity search (Simple, not using the model)
# query="..."
# retrieved_result=db.similarity_search(query)

# if retrieved_result:
#     print(retrieved_result[0].page_content)
# else:
#     print("Queried content not found!")
    
## Design a chatprompt template
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. Think step-by-step before providing a detailed answer.
    <context>
    {context}
    </context>

    Question: {input}
""")

## Model
from langchain_openai import AzureChatOpenAI

llm=AzureChatOpenAI(model="gpt-4o-mini")
# llm

## Create Chain
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain=create_stuff_documents_chain(llm, prompt)
# document_chain

## Using retriever

retriever=db.as_retriever()
# retriever

## Using retrieval chain
from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever,document_chain)

result=retrieval_chain.invoke({"input":"""Which word has been mentioned the most number of times in the resume?"""})
print(result.get("answer","answer not found"))