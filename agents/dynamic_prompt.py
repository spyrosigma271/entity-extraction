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

class GeneratedPrompt(BaseModel):
    role_context: str
    task_description: str
    output_format: str
    examples: str
    validation_rules: str

class WorkflowState(BaseModel):
    domains: List[Dict[str, Any]]
    entity_types: List[Dict[str, Any]]
    generated_prompt: GeneratedPrompt | None = None
    messages: List[Any] = Field(default_factory=list)
    error: str | None = None

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

prompt_generation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a prompt engineering expert. Create a single, comprehensive prompt for entity extraction that works across multiple domains. Format your response in clear text sections:

# Role and Context
[Define the AI's role and provide context]

# Task Description
[Clear description of what needs to be done]

# Output Format
[Specify how the output should be structured]

# Examples
[Provide relevant examples]

# Validation Rules
[List rules for valid inputs and outputs]

Consider:
1. How to handle multiple domains simultaneously
2. Required fields across all domains
3. Domain-specific validations
4. Cross-domain relationships
5. Edge cases and special handling

Include prompt engineering best practices:
1. Clear role definition
2. Explicit instructions
3. Step-by-step process
4. Consistent formatting
5. Error handling
6. Edge case coverage

Return only the prompt text with the above sections."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", """Generate a prompt for these domains:
{domains}

That can extract these entity types:
{entity_types}""")
])

prompt_refinement_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a prompt refinement expert. Review and enhance the generated prompt to ensure clarity, completeness, and effectiveness. Keep the same section structure:

# Role and Context
# Task Description
# Output Format
# Examples
# Validation Rules

Verify and enhance:
1. Clarity of instructions
2. Completeness of task description
3. Quality of examples
4. Thoroughness of validation rules
5. Cross-domain compatibility

Return the refined prompt text with the same sections."""),
    ("human", """Refine this prompt:
{current_prompt}

For these domains:
{domains}""")
])

def create_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)
    
    async def generate_prompt(state: WorkflowState) -> Dict[str, Any]:
        try:
            input_data = {
                "domains": json.dumps(state.domains, indent=2),
                "entity_types": json.dumps(state.entity_types, indent=2),
                "history": state.messages
            }
            
            messages = prompt_generation_prompt.format_messages(**input_data)
            response = await llm.agenerate([messages])
            
            # Parse the sectioned text response
            text = response.generations[0][0].text
            sections = {}
            current_section = None
            current_content = []
            
            for line in text.split('\n'):
                if line.startswith('# '):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                        current_content = []
                    current_section = line[2:].strip()
                elif current_section:
                    current_content.append(line)
                    
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            
            generated = GeneratedPrompt(
                role_context=sections.get('Role and Context', ''),
                task_description=sections.get('Task Description', ''),
                output_format=sections.get('Output Format', ''),
                examples=sections.get('Examples', ''),
                validation_rules=sections.get('Validation Rules', '')
            )
            
            state_dict = state.model_dump()
            state_dict.update({
                'generated_prompt': generated,
                'messages': [*state.messages, *messages, response.generations[0][0]]
            })
            return WorkflowState(**state_dict).model_dump()
            
        except Exception as e:
            state_dict = state.model_dump()
            state_dict['error'] = f"Prompt generation failed: {str(e)}\nLLM Response: {response.generations[0][0].text if 'response' in locals() else 'No response'}"
            return WorkflowState(**state_dict).model_dump()

    async def refine_prompt(state: WorkflowState) -> Dict[str, Any]:
        try:
            if state.error or not state.generated_prompt:
                return state.model_dump()
            
            # Convert prompt to sectioned text
            current_prompt = f"""# Role and Context
{state.generated_prompt.role_context}

# Task Description
{state.generated_prompt.task_description}

# Output Format
{state.generated_prompt.output_format}

# Examples
{state.generated_prompt.examples}

# Validation Rules
{state.generated_prompt.validation_rules}"""
                
            messages = prompt_refinement_prompt.format_messages(
                current_prompt=current_prompt,
                domains=json.dumps(state.domains, indent=2)
            )
            response = await llm.agenerate([messages])
            
            # Parse refined sectioned text
            text = response.generations[0][0].text
            sections = {}
            current_section = None
            current_content = []
            
            for line in text.split('\n'):
                if line.startswith('# '):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                        current_content = []
                    current_section = line[2:].strip()
                elif current_section:
                    current_content.append(line)
                    
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            
            refined = GeneratedPrompt(
                role_context=sections.get('Role and Context', ''),
                task_description=sections.get('Task Description', ''),
                output_format=sections.get('Output Format', ''),
                examples=sections.get('Examples', ''),
                validation_rules=sections.get('Validation Rules', '')
            )
            
            state_dict = state.model_dump()
            state_dict.update({
                'generated_prompt': refined,
                'messages': [*state.messages, *messages, response.generations[0][0]]
            })
            return WorkflowState(**state_dict).model_dump()
            
        except Exception as e:
            state_dict = state.model_dump()
            state_dict['error'] = f"Prompt refinement failed: {str(e)}\nLLM Response: {response.generations[0][0].text if 'response' in locals() else 'No response'}"
            return WorkflowState(**state_dict).model_dump()

    workflow.add_node("generate", generate_prompt)
    workflow.add_node("refine", refine_prompt)
    
    workflow.add_edge("generate", "refine")
    workflow.add_edge("refine", END)
    
    workflow.set_entry_point("generate")
    return workflow.compile()

async def generate_extraction_prompt(domains: List[Dict[str, Any]], entity_types: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        workflow = create_workflow()
        
        initial_state = WorkflowState(
            domains=domains,
            entity_types=entity_types
        ).model_dump()
        
        final_state = await workflow.ainvoke(initial_state)
        
        if final_state["error"]:
            return {
                "status": "error",
                "error": final_state["error"]
            }
            
        # Format final prompt as text
        prompt = final_state["generated_prompt"]
        final_prompt = f"""# Role and Context
{prompt["role_context"]}

# Task Description
{prompt["task_description"]}

# Output Format
{prompt["output_format"]}

# Examples
{prompt["examples"]}

# Validation Rules
{prompt["validation_rules"]}"""
            
        return {
            "status": "success",
            "prompt": final_prompt,
            "metadata": {
                "total_domains": len(domains),
                "total_entity_types": len(entity_types)
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
    
    test_domains = [
        {
            "name": "Medical",
            "confidence": 0.95,
            "reasoning": "Contains medical terminology and patient information",
            "key_indicators": ["patient", "diagnosis", "medications"]
        },
        {
            "name": "Insurance",
            "confidence": 0.75,
            "reasoning": "Contains billing and coverage information",
            "key_indicators": ["coverage", "policy", "claims"]
        }
    ]
    
    test_entity_types = [
        {
            "name": "PatientIdentifier",
            "description": "Information used to identify a patient",
            "properties": ["id_type", "value"],
            "examples": ["patient name", "medical record number"]
        },
        {
            "name": "ClinicalMeasurement",
            "description": "Medical measurements and vital signs",
            "properties": ["type", "value", "unit"],
            "examples": ["blood pressure", "temperature"]
        }
    ]
    
    result = asyncio.run(generate_extraction_prompt(test_domains, test_entity_types))
    print(json.dumps(result, indent=2))