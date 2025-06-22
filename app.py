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

# Load environment variables from a .env file (for local development)
load_dotenv()

# --- HEAVY LIFTING AT STARTUP ---
# These will be loaded once before workers are created thanks to --preload
print("Loading embeddings and connecting to Pinecone...")
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
print("Load complete.")
# --------------------------------

# --- LIGHTWEIGHT CONFIGURATION AT STARTUP ---
llm = OpenAI(temperature=0.4, max_tokens=500)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# ---------------------------------------------


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend API calls


# Health check
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "API is live 🔥"}), 200

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", "").strip()

        if not symptoms:
            return jsonify({"error": "Symptoms field is required"}), 400

        print(f"Received input: {symptoms}")

        response = rag_chain.invoke({"input": symptoms})
        answer = response["answer"]

        print(f"Generated answer: {answer}")

        return jsonify({"result": answer}), 200

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An internal server error occurred"}), 500


# This block is for running the app locally with `python app.py`
# It will be ignored by Gunicorn on Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)