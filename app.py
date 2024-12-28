import streamlit as st
import fitz  # We can also use pymupdf4llm 
import io
import re


def clean_text(text):
    """Clean and preprocess text"""
    # Convert to string if bytes
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters
    text = re.sub(r'[^\w\s.]', '', text)
    return text

def read_pdf(file):
    """Extract text from PDF file"""
    try:
        # Create PDF file stream
        pdf_stream = io.BytesIO(file.read())
        # Open PDF
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        text = ""
        # Extract text from all pages
        for page in doc:
            text += page.get_text()
        
        return clean_text(text)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def read_text(file):
    """Read text from txt file"""
    try:
        text = file.read()
        return clean_text(text)
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None


st.title('Entity Extraction')

file = st.file_uploader('Upload a file', type=['txt', 'pdf'])

if file:
    text = None
    if file.type == 'text/plain':
        text = read_text(file)
    elif file.type == 'application/pdf':
        text = read_pdf(file)
        
    if text:
        st.write("Processed Text:")
        st.write(text)