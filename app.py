import streamlit as st
import fitz  # We can also use pymupdf4llm
import io
import re
from dotenv import load_dotenv
import os , json
from groq import Groq

load_dotenv()

groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def clean_text(text):
    """Clean and preprocess text"""
    # Convert to string if bytes
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters
    text = re.sub(r"[^\w\s.]", "", text)
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


def extract_json_from_response(response_text):
    """Extract JSON from LLM response"""
    try:
        # Find JSON pattern between triple backticks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            # Parse JSON to validate and format
            return json.loads(json_str)
        
        # If no backticks, try to find direct JSON
        json_match = re.search(r'\{[^{]*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
            
        return None
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON: {str(e)}")
        return None


def extract_entities(schema, text):
    """Send schema and text to Groq-LLM and extract entities"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an entity extraction assistant. Your job is to identify and extract entities from "
                        "a given text based on a provided schema. The schema specifies the fields to extract and their descriptions. "
                        "Ensure that the output matches the schema structure and provides accurate results."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
            Below is the input document and the schema. Extract the entities from the text and format the results as a JSON object.
            
            <DOCUMENT>
            {text}
            </DOCUMENT>
            
            -----------------------
            
            <SCHEMA>
            {schema}
            </SCHEMA>
            
            Instructions:
            - Each schema field corresponds to an entity to extract.
            - Use the description in the schema to understand the context of each field.
            - Provide the results in JSON format, using schema field names as keys.
            - If a field cannot be extracted, set its value to null.
            - Return ONLY JSON object with extracted entities, nothing else.
        """,
                },
            ],
            model="llama3-70b-8192",
        )

        response = chat_completion.choices[0].message.content
        json_data = extract_json_from_response(response)
        
        if json_data:
            return json_data
        else:
            st.error("No valid JSON found in response")
            return None
    except Exception as e:
        st.error(f"Error extracting entities: {str(e)}")
        return None


st.title("Entity Extraction")

# File uploaders
text_file = st.file_uploader("Upload a document file (TXT or PDF)", type=["txt", "pdf"])

schema_file = st.file_uploader(
    "Upload a schema file (CSV or JSON)", type=["csv", "json"]
)

if schema_file and text_file:
    # Process schema
    schema_content = schema_file.read().decode("utf-8")

    # Process document
    document_text = None
    if text_file.type == "text/plain":
        document_text = read_text(text_file)
    elif text_file.type == "application/pdf":
        document_text = read_pdf(text_file)

    if schema_content and document_text:
        # Extract entities
        st.write("Schema Content:")
        st.write(schema_content)

        st.write("Document Text:")
        st.write(document_text)

        st.write("Extracting Entities...")
        extracted_entities = extract_entities(schema_content, document_text)

        if extracted_entities:
            st.write("Extracted Entities:")
            st.json(extracted_entities)
