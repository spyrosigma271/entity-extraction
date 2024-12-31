import streamlit as st
import fitz
import io
import json
import re
import asyncio
from typing import Dict, List, Any
from main import run_entity_extraction_pipeline  # Import the pipeline function
from streamlit_pdf_viewer import pdf_viewer
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

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
                "examples": field_info.get("examples", []),
            }
            entity_types.append(entity_type)

        return entity_types
    except json.JSONDecodeError:
        st.error("Invalid JSON schema format")
        return None

def convert_entities_to_annotations(entities, doc):
    """Convert extracted entities to PDF viewer annotation format"""
    annotations = []
    doc = fitz.open(stream=doc, filetype="pdf")
    
    for entity in entities:
        # Search for each entity text in the PDF
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_instances = page.search_for(entity['text'])
            
            # Create annotation for each instance found
            for inst in text_instances:
                annotations.append({
                    "page": page_num + 1,  # PDF pages are 1-based
                    "x": inst.x0,
                    "y": inst.y0,
                    "width": inst.width,
                    "height": inst.height,
                    "color": "yellow",  # You can map entity types to different colors
                    "text": entity['text']
                })
    
    return annotations

def main():
    st.title("Advanced Entity Extraction Pipeline")

    col1, col2 = st.columns(2)
    
    with col1:
        text_file = st.file_uploader("Upload a document file (TXT or PDF)", type=["txt", "pdf"])
    with col2:
        schema_file = st.file_uploader("Upload a schema file (JSON)", type=["json"])

    if schema_file and text_file:
        # Process schema
        schema_content = schema_file.read().decode("utf-8")
        entity_types = parse_schema(schema_content)

        if not entity_types:
            return

        # Create two columns for document and results
        doc_col, results_col = st.columns([0.4, 0.6])
        # Display document
        with doc_col:
            st.subheader("Document View")
            file_content = text_file.read()
            if text_file.type == "application/pdf":
                pdf_annotations = []
                
                # Initialize PDF viewer without annotations first
                # pdf_viewer_instance = pdf_viewer(
                #     file_content, 
                #     width=700, 
                #     height=700,
                #     annotations=pdf_annotations,
                #     scroll_to_annotation=True
                # )
                
                document_text = read_pdf(io.BytesIO(file_content))
            else:
                document_text = read_text(io.BytesIO(file_content))
                st.text_area("Text Content", document_text, height=400)

        # Display results
        with results_col:
            if document_text:
                st.subheader("Entity Types from Schema")
                with st.expander("Show entity types"):
                    st.json(entity_types)

                if st.button("Extract Entities"):
                    with st.spinner("Processing..."):
                        result = asyncio.run(
                            run_entity_extraction_pipeline(
                                text=document_text, 
                                initial_entity_types=entity_types
                            )
                        )

                        if result and result["status"] == "success":
                            st.success("Extraction completed!")

                            # Display results in tabs
                            tabs = st.tabs(["Domains", "Entity Types", "Raw Entities", "Standardized Entities"])
                            
                            with tabs[0]:
                                st.json(result["pipeline_results"]["domains"])
                            
                            with tabs[1]:
                                st.json(result["pipeline_results"]["entity_types"])
                            
                            with tabs[2]:
                                st.json(result["pipeline_results"]["raw_entities"])
                            
                            with tabs[3]:
                                st.json(result["pipeline_results"]["standardized_entities"])

                            with st.expander("Pipeline Metadata"):
                                st.json(result["metadata"] )
                            
                            def extract_entity_values(entities):
                                """Extract values from nested entity dictionaries"""
                                flat_entities = []
                                
                                for entity in entities:
                                    if 'value' in entity:
                                        # Extract all values from the value dictionary
                                        for key, value in entity['value'].items():
                                            if value and value != "Not Available":
                                                flat_entities.append({
                                                    'text': str(value),
                                                    'type': entity['type'],
                                                    'field': key
                                                })
                                
                                return flat_entities

                            # Convert extracted entities to annotations
                            standard_entities = result["pipeline_results"]["standardized_entities"]
                            flat_entities = extract_entity_values(standard_entities)
                            pdf_annotations = convert_entities_to_annotations(
                                flat_entities,
                                io.BytesIO(file_content)
                            )
                                  
                            # Update PDF viewer with annotations
                            with doc_col:
                                pdf_viewer(
                                    input=file_content,
                                    annotations=pdf_annotations,
                                    width=700,
                                    height=700,
                                    scroll_to_annotation=True
                                )
                            
                        else:
                            st.error(f"Extraction failed: {result.get('error', 'Unknown error')}")

            else:
                st.error("Error reading document file......")

if __name__ == "__main__":
    main()