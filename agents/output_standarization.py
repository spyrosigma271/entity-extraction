from typing import Dict, List, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.messages import SystemMessage
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class StandardizedEntity(BaseModel):
    id: str
    type: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    source: Dict[str, Any] = Field(default_factory=dict)

class WorkflowState(BaseModel):
    entities: List[Dict[str, Any]]
    domains: List[Dict[str, Any]]
    entity_types: List[Dict[str, Any]]
    standardized_entities: List[StandardizedEntity] = Field(default_factory=list)
    messages: List[Any] = Field(default_factory=list)
    error: str | None = None

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

standardization_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a data standardization expert. Convert extracted entities into a standardized format following these rules:

1. Each entity must have:
   - Unique 'id' (generated based on type and value)
   - 'type' matching defined entity types
   - 'value' in appropriate data type (string, number, boolean)
   - 'metadata' containing additional attributes
   - 'relationships' list with references to other entities
   - 'source' containing original extraction info

2. Standardize fields:
   - Convert dates to ISO format
   - Normalize units and measurements
   - Standardize names and identifiers
   - Structure nested objects consistently

Return the standardized entities in this exact JSON format:
{
    "standardized_entities": [
        {
            "id": "unique_identifier",
            "type": "entity_type",
            "value": "standardized_value",
            "metadata": {
                "confidence": float,
                "normalized_unit": "standard_unit",
                "other_relevant_fields": "values"
            },
            "relationships": [
                {
                    "type": "relationship_type",
                    "target_id": "related_entity_id",
                    "metadata": {}
                }
            ],
            "source": {
                "original_value": "raw_value",
                "extraction_confidence": float,
                "context": "original_context"
            }
        }
    ]
}

Only return valid JSON in exactly this format."""),
    ("human", """Standardize these entities:
{entities}

Using these entity types:
{entity_types}

Consider these domains:
{domains}""")
])

validation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a data validation expert. Review standardized entities to ensure:

1. All IDs are unique and properly formatted
2. Values match their declared types
3. Units are consistently normalized
4. Relationships are valid and reciprocal
5. Required metadata is present
6. Source information is preserved

Return the validated entities in the exact same JSON format, with any necessary corrections."""),
    ("human", """Validate these standardized entities:
{standardized_entities}

Using these entity types:
{entity_types}

Consider these domains:
{domains}""")
])

def create_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)
    
    async def standardize_entities(state: WorkflowState) -> Dict[str, Any]:
        try:
            messages = standardization_prompt.format_messages(
                entities=json.dumps(state.entities, indent=2),
                entity_types=json.dumps(state.entity_types, indent=2),
                domains=json.dumps(state.domains, indent=2)
            )
            response = await llm.agenerate([messages])
            
            result = json.loads(response.generations[0][0].text)
            standardized = [StandardizedEntity(**entity_dict) 
                          for entity_dict in result["standardized_entities"]]
            
            state_dict = state.model_dump()
            state_dict.update({
                'standardized_entities': standardized,
                'messages': [*state.messages, *messages, response.generations[0][0]]
            })
            return WorkflowState(**state_dict).model_dump()
            
        except Exception as e:
            state_dict = state.model_dump()
            state_dict['error'] = f"Standardization failed: {str(e)}\nLLM Response: {response.generations[0][0].text if 'response' in locals() else 'No response'}"
            return WorkflowState(**state_dict).model_dump()

    async def validate_standardized_entities(state: WorkflowState) -> Dict[str, Any]:
        try:
            if state.error or not state.standardized_entities:
                return state.model_dump()
                
            current_entities = {
                "standardized_entities": [entity.model_dump() for entity in state.standardized_entities]
            }
            
            messages = validation_prompt.format_messages(
                standardized_entities=json.dumps(current_entities, indent=2),
                entity_types=json.dumps(state.entity_types, indent=2),
                domains=json.dumps(state.domains, indent=2)
            )
            response = await llm.agenerate([messages])
            
            result = json.loads(response.generations[0][0].text)
            validated = [StandardizedEntity(**entity_dict) 
                        for entity_dict in result["standardized_entities"]]
            
            state_dict = state.model_dump()
            state_dict.update({
                'standardized_entities': validated,
                'messages': [*state.messages, *messages, response.generations[0][0]]
            })
            return WorkflowState(**state_dict).model_dump()
            
        except Exception as e:
            state_dict = state.model_dump()
            state_dict['error'] = f"Validation failed: {str(e)}\nLLM Response: {response.generations[0][0].text if 'response' in locals() else 'No response'}"
            return WorkflowState(**state_dict).model_dump()

    workflow.add_node("standardize", standardize_entities)
    workflow.add_node("validate", validate_standardized_entities)
    
    # workflow.add_edge("standardize", "validate")
    # workflow.add_edge("validate", END)
    workflow.add_edge("standardize", END)
    
    workflow.set_entry_point("standardize")
    return workflow.compile()

async def standardize_entity_data(
    entities: List[Dict[str, Any]],
    domains: List[Dict[str, Any]],
    entity_types: List[Dict[str, Any]]
) -> Dict[str, Any]:
    try:
        workflow = create_workflow()
        
        initial_state = WorkflowState(
            entities=entities,
            domains=domains,
            entity_types=entity_types
        ).model_dump()
        
        final_state = await workflow.ainvoke(initial_state)
        
        if final_state["error"]:
            return {
                "status": "error",
                "error": final_state["error"]
            }
            
        return {
            "status": "success",
            "standardized_entities": [
                entity
                for entity in final_state["standardized_entities"]
            ],
            "metadata": {
                "total_entities": len(final_state["standardized_entities"]),
                "unique_types": len(set(e["type"] for e in final_state["standardized_entities"])),
                "total_relationships": sum(len(e["relationships"]) for e in final_state["standardized_entities"])
            }
        }
            
    except Exception as e:
        return {
            "status": "error",
            "error": f"Workflow execution failed: {str(e)}"
        }

# Example usage
if __name__ == "__main__":
    import asyncio
    
    test_entities = [
        {
            "name": "John Smith",
            "type": "PatientIdentifier",
            "value": "John Smith",
            "confidence": 0.95,
            "reasoning": "Clearly identified as patient name in text",
            "context": "The patient, John Smith, was admitted",
            "relationships": []
        },
        {
            "name": "Blood Pressure",
            "type": "VitalSign",
            "value": "150/90 mmHg",
            "confidence": 0.98,
            "reasoning": "Standard BP measurement format",
            "context": "Initial vital signs showed BP 150/90 mmHg",
            "relationships": []
        }
    ]
    
    test_domains = [
        {
            "name": "Medical",
            "confidence": 0.95,
            "key_indicators": ["patient", "vital signs", "medications"]
        }
    ]
    
    test_entity_types = [
        {
            "name": "PatientIdentifier",
            "description": "Information used to identify a patient",
            "properties": ["id_type", "value"]
        },
        {
            "name": "VitalSign",
            "description": "Medical measurements",
            "properties": ["type", "value", "unit"]
        }
    ]
    
    result = asyncio.run(standardize_entity_data(
        entities=test_entities,
        domains=test_domains,
        entity_types=test_entity_types
    ))
    print(json.dumps(result, indent=2))