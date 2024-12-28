import streamlit as st
import fitz
import io
import re
import json
import pandas as pd
from typing import Optional, Dict, Any, Union
from groq import Groq
from dotenv import load_dotenv
import os


class EntityExtractor:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the entity extractor with optional API key"""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.api_key)

    def clean_text(self, text: Union[str, bytes]) -> str:
        """Clean and preprocess text"""
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        text = " ".join(text.split())
        text = re.sub(r"[^\w\s.]", "", text)
        return text

    def read_pdf(self, file) -> Optional[str]:
        """Extract text from PDF file"""
        try:
            pdf_stream = io.BytesIO(file.read())
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return self.clean_text(text)
        except Exception as e:
            return None

    def read_text(self, file) -> Optional[str]:
        """Read text from txt file"""
        try:
            text = file.read()
            return self.clean_text(text)
        except Exception as e:
            return None

    def extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from LLM response"""
        try:
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if json_match:
                return json.loads(json_match.group(1))

            json_match = re.search(r"\{[^{]*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            return None
        except json.JSONDecodeError:
            return None

    def extract_entities(self, text: str) -> Optional[Dict]:
        """Extract entities using Groq LLM"""
        try:
            chat_completion = self.groq_client.chat.completions.create(
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
            entity_name,entity_type,description,example
            full_name,text,Candidate's complete name,John Smith
            contact_email,text,Professional email address,john.smith@email.com
            phone_number,text,Contact phone number with country code,+1-555-123-4567
            location,text,Current city and country/state,New York NY USA
            linkedin_url,text,LinkedIn profile URL,linkedin.com/in/johnsmith
            work_experience,text,List of previous job positions and companies,Senior Developer at Tech Corp
            education,text,Academic degrees and institutions,MS Computer Science from MIT
            skills_technical,text,Technical skills and technologies,Python Java AWS Docker
            skills_soft,text,Soft skills and competencies,Team Leadership Project Management
            certifications,text,Professional certifications and licenses,AWS Certified Solutions Architect
            languages,text,Languages spoken and proficiency levels,English (Native) Spanish (Intermediate)
            projects,text,Notable projects completed,Built scalable cloud infrastructure
            achievements,text,Quantifiable accomplishments,Reduced costs by 30% Improved efficiency by 45%
            years_experience,number,Total years of professional experience,8
            current_role,text,Current or most recent job title,Senior Software Engineer
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
            return self.extract_json_from_response(response)
        except Exception:
            return None

    def process_file(self, file) -> Optional[str]:
        """Process uploaded file and extract text"""
        if file.type == "text/plain":
            return self.read_text(file)
        elif file.type == "application/pdf":
            return self.read_pdf(file)
        return None

    def process_schema(self, schema_file) -> Optional[Dict]:
        """Process schema file (CSV or JSON)"""
        try:
            if schema_file.type == "text/csv":
                df = pd.read_csv(schema_file)
                return df.to_dict(orient="records")
            elif schema_file.type == "application/json":
                return json.load(schema_file)
            return None
        except Exception:
            return None

    def extract(self, document_file) -> Optional[Dict]:
        """Main extraction workflow"""
        text = self.process_file(document_file)
        if not text:
            return None

        # schema = self.process_schema(schema_file)
        # if not schema:
        #     return None

        return self.extract_entities(text)


def run_streamlit_app():
    """Run the Streamlit UI"""
    st.title("Entity Extraction")

    text_file = st.file_uploader(
        "Upload a document file (TXT or PDF)", type=["txt", "pdf"]
    )
    # schema_file = st.file_uploader(
    #     "Upload a schema file (CSV or JSON)", type=["csv", "json"]
    # )

    if text_file:
        extractor = EntityExtractor()
        results = extractor.extract(text_file)
        if results:
            st.json(results)


if __name__ == "__main__":
    run_streamlit_app()
