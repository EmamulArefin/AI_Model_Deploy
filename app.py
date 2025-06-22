from flask import Flask, jsonify, request
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend API calls

# Load environment variables
load_dotenv()

# Set API keys from .env
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Load embedding model
embeddings = download_hugging_face_embeddings()

# Pinecone index name
index_name = "medicalbot"

# Load existing Pinecone vector index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Set up retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load OpenAI LLM
llm = OpenAI(temperature=0.4, max_tokens=500)

# Create prompt and chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Health check
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "API is live 🔥"}), 200

# Prediction endpoint for Next.js
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", "").strip()

        if not symptoms:
            return jsonify({"error": "Symptoms field is required"}), 400

        print("User Input:", symptoms)

        response = rag_chain.invoke({"input": symptoms})
        answer = response["answer"]

        print("Model Output:", answer)

        return jsonify({"result": answer}), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # fallback to 5000 locally
    app.run(host="0.0.0.0", port=port)

