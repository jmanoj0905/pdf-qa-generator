import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
import chromadb
import fitz
import io
from PIL import Image
import pytesseract
import shutil
import streamlit as st
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
import time
import dotenv


dotenv.load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.title("PDF Q&A Generator")
st.sidebar.header("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a file", type="pdf")

if uploaded_file is not None:
    start_time = time.time()
    st.success("Uploaded successfully!")
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.sidebar.success("PDF Uploaded Successfully!")
    st.sidebar.info("**Extracting text and images...**")
    
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
        
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"]))
            text += pytesseract.image_to_string(image) + "\n"
    
    st.sidebar.success("**Text Extraction Complete!**")
    st.sidebar.info("**Splitting into chunks...**")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    

    
    # vector_db = Chroma.from_documents(
    #     documents,
    #     embedding_model,
    #     persist_directory=db_path
    # )
    

    db_path = "/tmp/chroma_db"

    # Clean up existing database if it exists
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Create the directory
    os.makedirs(db_path, exist_ok=True)

    # Store embeddings into ChromaDB
    vector_db = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=db_path
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    st.sidebar.success("**Text Splitting and Embedding Complete!**")
    
    st.sidebar.info("**Generating questions**")
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key)
    query = "Generate a set of questions and answers based only on the given PDF content."
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    Generate a comprehensive set of high-quality questions and answers based ONLY on the provided text. **Maximize** the number of unique questions while ensuring technical depth.

    ### INSTRUCTIONS:
    1. **Generate at least 40 questions**, evenly distributed across three difficulty levels:
    - **1 Mark:** Factual recall and direct technical questions (15-20)
    - **2 Marks:** Conceptual understanding and application-based questions (10-15)
    - **4 Marks:** Deep analysis, real-world application, and critical thinking (10-15)

    2. **STRICT FORMAT:**
    - **1M:**  
        Q1) [Question]  
        A) [Answer]  
        Q2) [Question]  
        A) [Answer]  
    - **2M:**  
        Q1) [Question]  
        A) [Answer]  
    - **4M:**  
        Q1) [Question]  
        A) [Answer]  

    3. **CRITICAL RULES - MUST FOLLOW:**
    - Use **only** the substantive content from the text.
    - **Exclude** metadata, author details, document creation info, external references, or document title.
    - Never reference the document itself in any way.

    4. **QUESTION GENERATION STRATEGY:**
    - Extract every possible fact, number, date, term, or concept.
    - Frame multiple questions from each paragraph covering different angles.
    - Test relationships between concepts.
    - Create **scenario-based** and **"What if"** questions.
    - Include explanations of processes, hierarchies, and systems.
    - In **1M questions**, emphasize technical accuracy and definitions.
    - In **2M questions**, assess conceptual clarity with moderate application.
    - In **4M questions**, emphasize **practical application** and **deep understanding**.

    5. **ADDITIONAL REQUIREMENTS:**
    - Avoid redundant or repetitive questions.
    - Ensure concise yet complete answers.
    - **Prioritize technical questions** in **1M** while half them in **2M** are technical.
    - **For 4M questions, focus on in-depth application and reasoning.**

    ### TEXT:
    {context}
    """

    qa_res = llm.invoke(prompt)
    qa_text = qa_res.content
    qa_list = qa_text.split("\n")
    st.sidebar.success("**Question Generation Complete!**")
    end_time = time.time()
    st.sidebar.info(f"Time taken: {end_time - start_time:.2f} seconds")

    difficulty = None
    for line in qa_list:
        line = line.strip()
        
        if line.startswith("1M:"):
            difficulty = "1 Markers"
        elif line.startswith("2M:"):
            difficulty = "2 Markers"
        elif line.startswith("4M:"):
            difficulty = "4 Markers"

        if difficulty:
            print(f"\n{difficulty}\n" + "=" * len(difficulty))
            difficulty = None
        st.write(line)
else:
    st.warning("Please upload a PDF file.")
