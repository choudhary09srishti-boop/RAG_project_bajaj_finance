import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# -----------------------------
# Load FAISS index and chunks
# -----------------------------
index = faiss.read_index("faiss_index.bin")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# -----------------------------
# Load embedding model for searching
# -----------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Load Hugging Face QA model
# -----------------------------
qa_model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# -----------------------------
# Search function (top 3 chunks)
# -----------------------------
def search_chunks(query, top_k=3):
    q_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_vec, top_k)
    return [chunks[i] for i in indices[0]]

# -----------------------------
# Ask question
# -----------------------------
def ask_question(question):
    top_chunks = search_chunks(question)
    # Combine top chunks as context
    context = " ".join(top_chunks)
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']

# -----------------------------
# Test it
# -----------------------------
user_question = "What is the loan interest rate?"
answer = ask_question(user_question)
print("Answer:", answer)
