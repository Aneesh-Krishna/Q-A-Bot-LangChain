import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Setup
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # Enables cross-origin requests from your React frontend
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

retrieval_chain = None

@app.route("/", methods=["POST"])
def upload_file():
    global retrieval_chain

    uploaded_file = request.files.get("pdf_file")
    if not uploaded_file or not uploaded_file.filename.endswith(".pdf"):
        return jsonify({"success": False, "message": "Invalid or missing file."}), 400

    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    uploaded_file.save(file_path)

    # Load and process the document
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunk_docs = text_splitter.split_documents(docs)

    embeddings = AzureOpenAIEmbeddings(model="gpt-text-embedding-3-large")
    db = FAISS.from_documents(chunk_docs, embeddings)

    prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context. Think step-by-step before providing a detailed answer.
        <context>
        {context}
        </context>

        Question: {input}
    """)

    llm = AzureChatOpenAI(model="gpt-4o-mini")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return jsonify({"success": True, "message": "File uploaded and processed successfully."})

@app.route("/ask", methods=["POST"])
def ask_question():
    global retrieval_chain

    if retrieval_chain is None:
        return jsonify({"success": False, "answer": "No document uploaded yet."}), 400

    question = request.form.get("question", "")
    if not question.strip():
        return jsonify({"success": False, "answer": "Question is required."}), 400

    try:
        result = retrieval_chain.invoke({"input": question})
        answer = result.get("answer", "Answer not found.")
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        return jsonify({"success": False, "answer": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
