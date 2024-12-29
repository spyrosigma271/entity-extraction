from typing import Dict, List, Any
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

class EntityType(BaseModel):
    name: str
    description: str
    properties: List[str] = Field(default_factory=list)
    domain_specific: bool = False
    examples: List[str] = Field(default_factory=list)

class WorkflowState(BaseModel):
    domains: List[Dict[str, Any]]
    entity_types: List[EntityType] = Field(default_factory=list)
    type_relationships: List[Dict[str, Any]] = Field(default_factory=list)
    messages: List[Any] = Field(default_factory=list)
    depth: int = 0
    error: str | None = None

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

entity_type_discovery_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an entity type discovery assistant. Given a domain and existing entity types, reason about and identify additional relevant entity types. Return them in the following JSON format:
{
    "entity_types": [
        {
            "name": "type_name",
            "description": "detailed description of what this entity type represents",
            "properties": ["list", "of", "key", "properties", "this", "type", "should", "have"],
            "domain_specific": boolean indicating if this type is specific to the given domain,
            "examples": ["example1", "example2"]
        }
    ]
}

Think about:
1. What other entity types are commonly used alongside the given types?
2. What entity types are needed to support or validate the given types?
3. What entity types would make the data model more complete?
4. What entity types are required by industry standards?
5. What entity types are needed for compliance and regulation?

For example, in a medical domain:
If given PatientIdentifier type:
- Think about demographic types needed for patient records
- Consider insurance and billing related types
- Think about consent and authorization types
- Consider audit and access control types

Only return the JSON object, nothing else."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", """Given these entity types in the {domain} domain, what other types would typically be needed?

Current entity types:
{current_types}""")
])

relationship_type_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a type relationship and discovery assistant. Analyze relationships between entity types to identify new required or related types. Return them in the following JSON format:
{
    "type_relationships": [
        {
            "type": "relationship_type",
            "source_type": "source_entity_type",
            "target_type": "target_entity_type",
            "description": "description of how these types relate"
        }
    ],
    "new_entity_types": [
        {
            "name": "type_name",
            "description": "detailed description of what this entity type represents",
            "properties": ["list", "of", "key", "properties"],
            "domain_specific": boolean indicating if this is domain specific,
            "examples": ["example1", "example2"]
        }
    ]
}

For each existing type:
1. What types must it reference or be referenced by?
2. What types provide context or metadata for it?
3. What types are derived from or dependent on it?
4. What types are needed to track its history or changes?
5. What types support its validation or verification?

Use domain knowledge to identify missing but necessary types.
Only return the JSON object, nothing else."""),
    ("human", """Based on the relationships between these entity types in the {domain} domain, what additional types are needed?

Current entity types:
{current_types}""")
])

def create_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)
    
    async def discover_entity_types(state: WorkflowState) -> Dict[str, Any]:
        try:
            domain = state.domains[0]["name"] if state.domains else "General"
            
            input_data = {
                "domain": domain,
                "current_types": json.dumps([type.model_dump() for type in state.entity_types], indent=2),
                "history": state.messages
            }
            
            messages = entity_type_discovery_prompt.format_messages(**input_data)
            response = await llm.agenerate([messages])
            
            new_types = [
                EntityType(**type_dict)
                for type_dict in json.loads(response.generations[0][0].text)["entity_types"]
            ]
            
            state_dict = state.model_dump()
            state_dict.update({
                'entity_types': [*state.entity_types, *new_types],
                'messages': [*state.messages, *messages, response.generations[0][0]]
            })
            return WorkflowState(**state_dict).model_dump()
            
        except Exception as e:
            state_dict = state.model_dump()
            state_dict['error'] = f"Entity type discovery failed: {str(e)}\nLLM Response: {response.generations[0][0].text if 'response' in locals() else 'No response'}"
            return WorkflowState(**state_dict).model_dump()

    async def analyze_type_relationships(state: WorkflowState) -> Dict[str, Any]:
        try:
            domain = state.domains[0]["name"] if state.domains else "General"
            formatted_types = json.dumps([type.model_dump() for type in state.entity_types], indent=2)
            
            messages = relationship_type_prompt.format_messages(
                domain=domain,
                current_types=formatted_types
            )
            response = await llm.agenerate([messages])
            
            result = json.loads(response.generations[0][0].text)
            relationships = result.get("type_relationships", [])
            new_types = [
                EntityType(**type_dict)
                for type_dict in result.get("new_entity_types", [])
            ]
            
            state_dict = state.model_dump()
            state_dict.update({
                'type_relationships': [*state.type_relationships, *relationships],
                'entity_types': [*state.entity_types, *new_types],
                'messages': [*state.messages, *messages, response.generations[0][0]],
                'depth': state.depth + 1
            })
            return WorkflowState(**state_dict).model_dump()
            
        except Exception as e:
            state_dict = state.model_dump()
            state_dict['error'] = f"Type relationship analysis failed: {str(e)}\nLLM Response: {response.generations[0][0].text if 'response' in locals() else 'No response'}"
            return WorkflowState(**state_dict).model_dump()

    def should_continue(state: WorkflowState):
        should_continue_flag = (
            not state.error and
            state.depth < 1 and
            len(state.entity_types) > len(state.type_relationships)
        )
        return "continue" if should_continue_flag else "end"

    workflow.add_node("discover", discover_entity_types)
    workflow.add_node("analyze", analyze_type_relationships)

    workflow.add_edge("discover", "analyze")
    
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "continue": "analyze",
            "end": END
        }
    )

    workflow.set_entry_point("discover")
    return workflow.compile()

async def run_type_discovery(domains: List[Dict[str, Any]], initial_types: List[Dict[str, Any]] = None):
    try:
        workflow = create_workflow()
        
        validated_initial_types = [
            EntityType(**type_dict) for type_dict in (initial_types or [])
        ]
        
        initial_state = WorkflowState(
            domains=domains,
            entity_types=validated_initial_types
        ).model_dump()
        
        final_state = await workflow.ainvoke(initial_state)
        
        return {
            "status": "success",
            "entity_types": [type for type in final_state["entity_types"]],
            "type_relationships": final_state["type_relationships"],
            "metadata": {
                "depth_reached": final_state["depth"],
                "total_types": len(final_state["entity_types"]),
                "total_relationships": len(final_state["type_relationships"])
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
    
    test_domains = [{"name": "Medical", "confidence": 0.95}]
    test_types = [
        {
            "name": "PatientIdentifier",
            "description": "Types of information used to identify a patient in medical records",
            "properties": ["id_type", "value", "issuance_date"],
            "domain_specific": True,
            "examples": ["patient name", "medical record number", "social security number"]
        },
        {
            "name": "VitalSign",
            "description": "Types of physiological measurements taken to assess basic body functions",
            "properties": ["measurement_type", "value", "unit", "timestamp"],
            "domain_specific": True,
            "examples": ["blood pressure", "heart rate", "temperature", "respiratory rate"]
        }
    ]
    
    result = asyncio.run(run_type_discovery(test_domains, test_types))
    print(json.dumps(result, indent=2))