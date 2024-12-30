from typing import Dict, List, Any, Union
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class Entity(BaseModel):
    name: str
    type: str
    value: Union[str, Dict[str, Any]]  # Allow either string or dictionary value
    confidence: float
    reasoning: str
    context: str | None = None
    relationships: List[Dict[str, Any]] = Field(default_factory=list)


class WorkflowState(BaseModel):
    text: str
    extraction_prompt: str
    domains: List[Dict[str, Any]]
    entity_types: List[Dict[str, Any]]
    entities: List[Entity] = Field(default_factory=list)
    messages: List[Any] = Field(default_factory=list)
    depth: int = 0
    error: str | None = None

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# First pass: Initial extraction with reasoning
initial_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """{extraction_prompt}

Additional Instructions for Reasoning:
1. For each entity, explain your reasoning process
2. Consider context and relationships between entities
3. Handle ambiguous cases carefully
4. Assess confidence based on available evidence
5. Document any assumptions made

Return your analysis in this format:
{{
    "entities": [
        {{
            "name": "entity name",
            "type": "entity type",
            "value": "extracted value",
            "confidence": float between 0-1,
            "reasoning": "explanation of why this is an entity and how it was determined",
            "context": "relevant surrounding context",
            "relationships": [
                {{
                    "type": "relationship type",
                    "target": "related entity",
                    "description": "relationship description"
                }}
            ]
        }}
    ],
    "analysis": "overall analysis of the extraction process and key decisions made"
}}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", """Extract entities from this text:
{text}

Using these entity types:
{entity_types}

Consider these domains:
{domains}""")
])

validation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an entity validation expert. Review the extracted entities and:
1. Check for missed entities
2. Validate entity types and relationships
3. Resolve any ambiguities
4. Merge duplicate entities
5. Refine confidence scores
6. Enhance reasoning explanations

Return your validation in the following exact JSON format:
{{
    "entities": [
        {{
            "name": "entity name",
            "type": "entity type",
            "value": "extracted value",
            "confidence": float between 0-1,
            "reasoning": "explanation of why this is an entity and how it was determined",
            "context": "relevant surrounding context",
            "relationships": [
                {{
                    "type": "relationship type",
                    "target": "related entity",
                    "description": "relationship description"
                }}
            ]
        }}
    ],
    "analysis": "overall analysis of validation and refinements made"
}}

IMPORTANT:
1. Ensure the response is valid JSON
2. Include all required fields for each entity
3. Use double quotes for strings
4. Use numbers for confidence scores
5. Only return the JSON object, no additional text or explanations
6. Maintain the exact structure shown above"""),
    ("human", """Review these extracted entities:
{current_entities}

From this text:
{text}

Using these entity types:
{entity_types}

Consider these domains:
{domains}

Remember to return only valid JSON following the exact format specified.""")
])

def create_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)
    
    async def extract_entities(state: WorkflowState) -> Dict[str, Any]:
        try:
            input_data = {
                "extraction_prompt": state.extraction_prompt,
                "text": state.text,
                "entity_types": json.dumps(state.entity_types, indent=2),
                "domains": json.dumps(state.domains, indent=2),
                "history": state.messages
            }
            
            messages = initial_extraction_prompt.format_messages(**input_data)
            response = await llm.agenerate([messages])
            
            result = json.loads(response.generations[0][0].text)
            new_entities = [Entity(**entity_dict) for entity_dict in result["entities"]]
            
            state_dict = state.model_dump()
            state_dict.update({
                'entities': new_entities,
                'messages': [*state.messages, *messages, response.generations[0][0]]
            })
            return WorkflowState(**state_dict).model_dump()
            
        except Exception as e:
            state_dict = state.model_dump()
            state_dict['error'] = f"Entity extraction failed: {str(e)}\nLLM Response: {response.generations[0][0].text if 'response' in locals() else 'No response'}"
            return WorkflowState(**state_dict).model_dump()

    async def validate_entities(state: WorkflowState) -> Dict[str, Any]:
        # try:
        if state.error or not state.entities:
            return state.model_dump()
            
        current_entities = {
            "entities": [entity.model_dump() for entity in state.entities]
        }
        
        messages = validation_prompt.format_messages(
            current_entities=json.dumps(current_entities, indent=2),
            text=state.text,
            entity_types=json.dumps(state.entity_types, indent=2),
            domains=json.dumps(state.domains, indent=2)
        )
        response = await llm.agenerate([messages])
        result = json.loads(response.generations[0][0].text)
        validated_entities = [Entity(**entity_dict) for entity_dict in result["entities"]]
        
        state_dict = state.model_dump()
        state_dict.update({
            'entities': validated_entities,
            'messages': [*state.messages, *messages, response.generations[0][0]],
            'depth': state.depth + 1
        })
        return WorkflowState(**state_dict).model_dump()
            
        # except Exception as e:
        #     state_dict = state.model_dump()
        #     state_dict['error'] = f"Entity validation failed: {str(e)}\nLLM Response: {response.generations[0][0].text if 'response' in locals() else 'No response'}"
        #     return WorkflowState(**state_dict).model_dump()

    def should_continue(state: WorkflowState):
        # Continue if there are significant changes in entities or confidence scores
        should_continue_flag = (
            not state.error and
            state.depth < 2  # Limit refinement iterations
        )
        return "continue" if should_continue_flag else "end"

    workflow.add_node("extract", extract_entities)
    workflow.add_node("validate", validate_entities)
    
    workflow.add_edge("extract", "validate")
    
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "continue": "validate",
            "end": END
        }
    )
    
    workflow.set_entry_point("extract")
    return workflow.compile()

async def extract_entities(
    text: str,
    extraction_prompt: str,
    domains: List[Dict[str, Any]], 
    entity_types: List[Dict[str, Any]]
) -> Dict[str, Any]:
    try:
        workflow = create_workflow()
        
        initial_state = WorkflowState(
            text=text,
            extraction_prompt=extraction_prompt,
            domains=domains,
            entity_types=entity_types
        ).model_dump()
        
        final_state = await workflow.ainvoke(initial_state)
        
        if final_state["error"]:
            return {
                "status": "error",
                "error": final_state["error"]
            }
        
        # Sort entities by confidence
        entities = sorted(
            final_state["entities"],
            key=lambda x: x["confidence"],
            reverse=True
        )
        
        return {
            "status": "success",
            "entities": [entity for entity in entities],
            "metadata": {
                "depth_reached": final_state["depth"],
                "total_entities": len(entities),
                "average_confidence": sum(e["confidence"] for e in entities) / len(entities) if entities else 0
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
    
    test_text = """
    The patient, John Smith, was admitted on January 15, 2024, with severe chest pain.
    Initial vital signs showed BP 150/90 mmHg. Medical history includes hypertension
    managed with lisinopril 10mg daily.
    """
    
    test_prompt = """# Role and Context
You are an entity extraction expert specialized in medical and clinical documentation.

# Task Description
Extract and analyze entities from medical texts, considering patient information, measurements, and clinical findings.

# Output Format
Provide structured entity extraction with confidence scores and reasoning.

# Examples
Input: "Patient John Doe's BP was 120/80"
Output: Identifies "John Doe" as PatientIdentifier and "120/80" as VitalSign

# Validation Rules
1. All patient identifiers must be verified
2. Measurements must include units
3. Dates must be standardized"""

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
    
    result = asyncio.run(extract_entities(
        text=test_text,
        extraction_prompt=test_prompt,
        domains=test_domains,
        entity_types=test_entity_types
    ))
    print(json.dumps(result, indent=2))