from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv

load_dotenv()

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

result=retrieval_chain.invoke({"input":"""Update the resume to get the ATS score more than 90% and tell how much ATS score(exact score) does this new resume give. JD: Job Title: Junior Software Developer (Fresher)

Location: [Your Company Location]Job Type: Full-timeExperience: FresherSalary: [As per industry standards]

Job Summary:

We are looking for a highly motivated and enthusiastic Junior Software Developer to join our dynamic team. As a fresher, you will have the opportunity to work on real-world projects and enhance your development skills. You will be involved in designing, developing, and maintaining applications using .NET technologies while following best practices in Object-Oriented Programming (OOP) and version control using Git.

If you have a passion for software development and a willingness to learn new technologies, we encourage you to apply!

Key Responsibilities:

Develop, test, and maintain software applications using .NET framework.

Implement OOP principles to write clean, reusable, and scalable code.

Collaborate with senior developers and participate in code reviews.

Use Git for version control and ensure code is properly maintained.

Troubleshoot and debug applications to enhance performance.

Stay updated with the latest industry trends and technologies.

Required Skills:

Proficiency in .NET (C#, ASP.NET, .NET Core).

Strong understanding of Object-Oriented Programming (OOP) concepts.

Familiarity with Git and version control workflows.

Basic knowledge of SQL databases.

Strong problem-solving and analytical skills.

Good communication and teamwork abilities.

Preferred / Add-on Skills (Good to Have):

Knowledge of React.js for front-end development.

Understanding of cloud platforms such as AWS or Azure.

Exposure to RESTful APIs and web services.

Educational Qualification:

Bachelor’s degree in Computer Science, Information Technology, or a related field.

Why Join Us?

Work on exciting and innovative projects.

Learn from experienced professionals in the industry.

Friendly and supportive work environment.

Opportunities for career growth and skill development.

If you are eager to kick-start your career in software development, apply now and become a part of our growing team!

How to Apply:Send your updated resume to [Your Email/Company Career Page] with the subject “Application for Junior Software Developer (Fresher)”."""})
print(result.get("answer","answer not found"))