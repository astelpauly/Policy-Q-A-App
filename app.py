import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from dotenv import load_dotenv


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") 
MODEL_NAME = "openai/gpt-oss-20b"


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def get_hf_client():
    if not HF_TOKEN:
        st.error("Hugging Face token not found. Please set HF_TOKEN.")
        st.stop()
    return InferenceClient(model=MODEL_NAME, token=HF_TOKEN)

hf_client = get_hf_client()


def hf_llm(prompt):
    messages = [{"role": "user", "content": prompt}]
    output = hf_client.chat_completion(messages, max_tokens=1024)
    return output.choices[0].message["content"]


def get_answer(query):
    vector_store = load_vector_store()

    docs = vector_store.similarity_search(query, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful assistant. Use the context below to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    return hf_llm(prompt)


st.title("📄 Policy Q&A")
st.write("Ask a question about the PDF and get an answer.")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Getting answers..."):
            answer = get_answer(query)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")

