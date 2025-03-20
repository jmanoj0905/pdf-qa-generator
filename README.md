# PDF Q&A Generator

PDF Q&A Generator extracts text from PDFs, generates vector embeddings, and formulates questions and answers using Llama 3.3-70B.

## Features

- Extracts text and images from PDFs using PyMuPDF and Tesseract OCR.
- Splits text into chunks and stores them as embeddings in a Chroma vector database.
- Uses LangChain and Groq API to generate structured Q&A sets.
- Provides a Streamlit-based UI for user interaction.

## Requirements

- Python 3.9 or higher
- Pip
- A Groq API key (stored in a `.env` file)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/pdf-qa-generator.git
   cd pdf-qa-generator
2. Create a venv
3. Install dependencies
   ```sh
   pip install -r requirements.txt
4. Run using streamlit
   ```sh
   streamlit run app.py
