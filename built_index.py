import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf(file_path, start_page=5, end_page=19):
    reader = PdfReader(file_path)
    text = ""
    for i in range(start_page - 1, end_page):
        page_text = reader.pages[i].extract_text()
        if page_text:
            text += page_text + "\n"
    pattern = r"#add your header and footer pattern here"
    clean_text = re.sub(pattern, "", text, flags=re.MULTILINE)
    return clean_text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

folder_path = "data/faiss_index"

def create_vector_store(cleaned_text):
    chunks = chunk_text(cleaned_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(folder_path)
    print(f"✅ Vector store saved at {folder_path}")

if __name__ == "__main__":
    pdf_path = os.path.join("data", "Add your pdf file folder path")
    text = load_pdf(pdf_path) 
    print("📄 PDF loaded.")
    create_vector_store(text)
    print("🧹 Creating vector store.")
