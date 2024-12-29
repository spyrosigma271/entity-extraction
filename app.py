import streamlit as st
import fitz
import io
import json
import re
import asyncio
from typing import Dict, List, Any
from main import run_entity_extraction_pipeline  # Import the pipeline function

def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    text = " ".join(text.split())
    text = re.sub(r"[^\w\s.]", "", text)
    return text

def read_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_stream = io.BytesIO(file.read())
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return clean_text(text)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def read_text(file) -> str:
    """Read text from txt file"""
    try:
        text = file.read()
        return clean_text(text)
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

def parse_schema(schema_content: str) -> List[Dict[str, Any]]:
    """Parse schema file content into entity types"""
    try:
        schema_data = json.loads(schema_content)
        entity_types = []
        
        for field_name, field_info in schema_data.items():
            entity_type = {
                "name": field_name,
                "description": field_info.get("description", ""),
                "properties": field_info.get("properties", []),
                "domain_specific": field_info.get("domain_specific", False),
                "examples": field_info.get("examples", [])
            }
            entity_types.append(entity_type)
            
        return entity_types
    except json.JSONDecodeError:
        st.error("Invalid JSON schema format")
        return None

def main():
    st.title("Advanced Entity Extraction Pipeline")
    
    # File uploaders
    text_file = st.file_uploader("Upload a document file (TXT or PDF)", type=["txt", "pdf"])
    schema_file = st.file_uploader("Upload a schema file (JSON)", type=["json"])
    
    if schema_file and text_file:
        # Process schema
        schema_content = schema_file.read().decode("utf-8")
        entity_types = parse_schema(schema_content)
        
        if not entity_types:
            return
            
        # Process document
        document_text = None
        if text_file.type == "text/plain":
            document_text = read_text(text_file)
        elif text_file.type == "application/pdf":
            document_text = read_pdf(text_file)
            
        if document_text:
            st.subheader("Document Preview")
            with st.expander("Show document text"):
                st.text(document_text)
                
            st.subheader("Entity Types from Schema")
            with st.expander("Show entity types"):
                st.json(entity_types)
            
            if st.button("Extract Entities"):
                with st.spinner("Processing..."):
                    # Run the pipeline from main.py
                    result = asyncio.run(run_entity_extraction_pipeline(
                        text=document_text,
                        initial_entity_types=entity_types
                    ))
                    
                    if result and result["status"] == "success":
                        st.success("Extraction completed!")
                        
                        # Display domains
                        with st.expander("Identified Domains"):
                            st.json(result["pipeline_results"]["domains"])
                        
                        # Display all entity types
                        with st.expander("Entity Types (Original + Discovered)"):
                            st.json(result["pipeline_results"]["entity_types"])
                        
                        # Display raw extracted entities
                        with st.expander("Raw Extracted Entities"):
                            st.json(result["pipeline_results"]["raw_entities"])
                        
                        # Display standardized entities
                        st.subheader("Standardized Entities")
                        st.json(result["pipeline_results"]["standardized_entities"])
                        
                        # Display metadata
                        with st.expander("Pipeline Metadata"):
                            st.json(result["metadata"])
                    else:
                        st.error(f"Extraction failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()