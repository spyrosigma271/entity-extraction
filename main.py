import asyncio
from typing import Dict, List, Any
import json
from dotenv import load_dotenv
import os

# Import all the workflow functions from the modules
from agents.domain_identifier import identify_domains
from agents.dynamic_prompt import generate_extraction_prompt
from agents.entity_expansion import run_type_discovery
from agents.entity_extractor import extract_entities
from agents.output_standarization import standardize_entity_data  # New import

import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

async def run_entity_extraction_pipeline(text: str, initial_entity_types: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        print("Step 1: Identifying domains...")
        domains_result = await identify_domains(text)
        if domains_result["status"] != "success":
            return {"status": "error", "error": f"Domain identification failed: {domains_result.get('error')}"}
        
        domains = domains_result["domains"]
        print(f"Found {len(domains)} domains")
        
        print("\nStep 2: Discovering entity types...")
        types_result = await run_type_discovery(domains, initial_entity_types)
        if types_result["status"] != "success":
            return {"status": "error", "error": f"Entity type discovery failed: {types_result.get('error')}"}
        
        entity_types = types_result["entity_types"]
        print(f"Discovered {len(entity_types)} entity types")
        
        print("\nStep 3: Generating extraction prompt...")
        prompt_result = await generate_extraction_prompt(domains, entity_types)
        if prompt_result["status"] != "success":
            return {"status": "error", "error": f"Prompt generation failed: {prompt_result.get('error')}"}
        
        extraction_prompt = prompt_result["prompt"]
        print("Generated extraction prompt")
        
        print("\nStep 4: Extracting entities...")
        entities_result = await extract_entities(
            text=text,
            extraction_prompt=extraction_prompt,
            domains=domains,
            entity_types=entity_types
        )
        if entities_result["status"] != "success":
            return {"status": "error", "error": f"Entity extraction failed: {entities_result.get('error')}"}
        
        print(f"Extracted {len(entities_result['entities'])} entities")

        print("\nStep 5: Standardizing entity data...")
        standard_result = await standardize_entity_data(
            entities=entities_result["entities"],
            domains=domains,
            entity_types=entity_types
        )
        if standard_result["status"] != "success":
            return {"status": "error", "error": f"Entity standardization failed: {standard_result.get('error')}"}
        
        print(f"Standardized {len(standard_result['standardized_entities'])} entities")
        
        return {
            "status": "success",
            "pipeline_results": {
                "domains": domains,
                "entity_types": entity_types,
                "extraction_prompt": extraction_prompt,
                "raw_entities": entities_result["entities"],
                "standardized_entities": standard_result["standardized_entities"]
            },
            "metadata": {
                "domains_metadata": domains_result.get("metadata", {}),
                "types_metadata": types_result.get("metadata", {}),
                "prompt_metadata": prompt_result.get("metadata", {}),
                "extraction_metadata": entities_result.get("metadata", {}),
                "standardization_metadata": standard_result.get("metadata", {})
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Pipeline execution failed: {str(e)}"
        }

async def main():
    # Example usage
    test_text = """
    The patient presented with elevated blood pressure (150/90 mmHg) and reported 
    chest pain radiating to the left arm. ECG showed ST-segment elevation in leads 
    V1-V4. Initial troponin levels were elevated at 0.5 ng/mL. Treatment with 
    aspirin and nitroglycerin was initiated immediately.
    """
    
    # Optional: Provide initial entity types to guide the discovery
    initial_types = [
        {
            "name": "PatientSymptom",
            "description": "Clinical symptoms reported by or observed in a patient",
            "properties": ["symptom_type", "severity", "location", "onset"],
            "domain_specific": True,
            "examples": ["chest pain", "shortness of breath", "fever"]
        }
    ]
    
    result = await run_entity_extraction_pipeline(test_text, initial_types)
    print("\nFinal Pipeline Results:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())