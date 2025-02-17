import os
import faiss
import json
import tempfile
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

# Groq API Key (Set this in your environment variables)
GROQ_API_KEY = os.environ["GROQ_API_KEY"]  # Replace with your actual key
client = Groq(api_key=GROQ_API_KEY)


# Initialize SentenceTransformer for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to create chunks from text
def create_chunks(text, chunk_size=300):
    sentences = text.split(".")
    chunks = []
    chunk = []
    length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        length += len(sentence.split())
        if length <= chunk_size:
            chunk.append(sentence)
        else:
            chunks.append(". ".join(chunk) + ".")
            chunk = [sentence]
            length = len(sentence.split())
    if chunk:
        chunks.append(". ".join(chunk) + ".")
    return chunks

# Function to create embeddings and store them in FAISS
def store_chunks_in_faiss(chunks):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Function to search the most relevant chunks from FAISS
def search_faiss(query, chunks, index, embeddings):
    query_vector = embedder.encode([query])
    _, indices = index.search(query_vector, 1)
    return chunks[indices[0][0]]

# Function to interact with Groq API
def ask_groq(question):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": question}],
        model="llama-3.3-70b-versatile"
    )
    return chat_completion.choices[0].message.content

# Streamlit frontend
def main():
    st.title("📄 RAG-Based PDF Question Answering App")
    st.write("Upload a PDF document, and ask questions based on its content.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from PDF
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            pdf_path = temp_pdf.name
        text = extract_text_from_pdf(pdf_path)
        
        # Create chunks and store them in FAISS
        chunks = create_chunks(text)
        index, embeddings = store_chunks_in_faiss(chunks)
        
        st.success("✅ PDF processed and indexed successfully!")
        
        # Question input
        user_query = st.text_input("Ask a question about the document:")
        
        if user_query:
            # Retrieve relevant chunk
            relevant_chunk = search_faiss(user_query, chunks, index, embeddings)
            st.write(f"🔍 Relevant chunk: {relevant_chunk}")
            
            # Query Groq with the relevant chunk
            groq_response = ask_groq(f"Context: {relevant_chunk}\nQuestion: {user_query}")
            st.write(f"🤖 Groq's Answer: {groq_response}")

if __name__ == "__main__":
    main()
