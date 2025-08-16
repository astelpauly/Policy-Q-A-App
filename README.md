Policy Q&A with LangChain & Hugging Face

This project is a Streamlit-based web application that allows users to ask questions about a PDF document and get answers using LangChain, FAISS vector store, and Hugging Face inference models.

The app reads a PDF, creates embeddings (vectors), stores them in a FAISS index, and uses an LLM to answer questions based on the relevant content.

How to Run This Project

1. Clone the repository:
git clone https://github.com/astelpauly/Policy-Q-A-App.git
cd "Policy-Q-A-App"

2. Create a virtual environment:
python -m venv venv
Activate it.

3. Install required packages:
-> pip install -r requirements.txt

4. Create Vectors:
   Add your PDF file to the data/ folder.
   Make sure the file path in built_index.py points to your PDF.
   If your PDF contains headers, footers, or page numbers, clean them before running the index.
   Run the script to build the FAISS index:
   -> python built_index.py

   Vectors will be created in data/faiss_index.

6. Configure Secrets
Create a .env file in the project root.
Add your Hugging Face inference API key:

HF_TOKEN=your_huggingface_token

6. Run the Streamlit App:
-> streamlit run app.py
The app will open in your default browser.
Enter a question about your PDF and get an answer from the LLM.

Project Structure:
project-root/
│
├── app.py             # Main Streamlit app
├── built_index.py     # Script to create FAISS vectors from PDF
├── requirements.txt   # Python dependencies
├── .env               # Environment variables (API keys)
├── data/              # PDFs and FAISS indexes
│   └── faiss_index/   # Stored embeddings
├── .gitignore         # Files/folders to ignore
└── README.md          # Project documentation


Notes:
Ensure the virtual environment (venv) is ignored in Git.
Always clean the PDF headers/footers to improve vector quality.
