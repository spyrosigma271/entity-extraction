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

class Domain(BaseModel):
    name: str
    confidence: float
    reasoning: str
    key_indicators: List[str]

class WorkflowState(BaseModel):
    text: str
    domains: List[Domain] = Field(default_factory=list)
    messages: List[Any] = Field(default_factory=list)
    error: str | None = None

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

domain_analysis_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a domain identification expert. Analyze the text and identify potential domains it belongs to. Return your analysis in JSON format:
{
    "domains": [
        {
            "name": "domain name",
            "confidence": float between 0-1,
            "reasoning": "explanation for this domain",
            "key_indicators": ["list", "of", "supporting", "evidence"]
        }
    ]
}

Consider:
1. Technical terminology and jargon
2. Common patterns and structures
3. Subject matter context
4. Professional or industry-specific language

Identify both primary and related domains. For example, a medical text might have:
- Primary: Clinical Medicine (0.95 confidence)
- Related: Healthcare Administration (0.7 confidence)
- Related: Medical Insurance (0.5 confidence)

Only return the JSON object, nothing else."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Identify domains for this text:\n\n{text}")
])

domain_refinement_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a domain refinement expert. Review and refine the initial domain analysis to:
1. Resolve domain overlaps
2. Adjust confidence scores
3. Merge similar domains
4. Ensure confidence scores are well-calibrated

Return the refined analysis in the same JSON format:
{
    "domains": [
        {
            "name": "domain name",
            "confidence": float between 0-1,
            "reasoning": "explanation for this domain",
            "key_indicators": ["list", "of", "supporting", "evidence"]
        }
    ]
}

Only return the JSON object, nothing else."""),
    ("human", """Review and refine this domain analysis:

{current_analysis}""")
])

def create_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)
    
    async def analyze_domains(state: WorkflowState) -> Dict[str, Any]:
        try:
            input_data = {
                "text": state.text,
                "history": state.messages
            }
            
            messages = domain_analysis_prompt.format_messages(**input_data)
            response = await llm.agenerate([messages])
            
            new_domains = [
                Domain(**domain_dict)
                for domain_dict in json.loads(response.generations[0][0].text)["domains"]
            ]
            
            state_dict = state.model_dump()
            state_dict.update({
                'domains': new_domains,
                'messages': [*state.messages, *messages, response.generations[0][0]]
            })
            return WorkflowState(**state_dict).model_dump()
            
        except Exception as e:
            state_dict = state.model_dump()
            state_dict['error'] = f"Domain analysis failed: {str(e)}\nLLM Response: {response.generations[0][0].text if 'response' in locals() else 'No response'}"
            return WorkflowState(**state_dict).model_dump()

    async def refine_domains(state: WorkflowState) -> Dict[str, Any]:
        try:
            if state.error:
                return state.model_dump()
                
            current_analysis = {
                "domains": [domain.model_dump() for domain in state.domains]
            }
            
            messages = domain_refinement_prompt.format_messages(
                current_analysis=json.dumps(current_analysis, indent=2)
            )
            response = await llm.agenerate([messages])
            
            refined_domains = [
                Domain(**domain_dict)
                for domain_dict in json.loads(response.generations[0][0].text)["domains"]
            ]
            
            state_dict = state.model_dump()
            state_dict.update({
                'domains': refined_domains,
                'messages': [*state.messages, *messages, response.generations[0][0]]
            })
            return WorkflowState(**state_dict).model_dump()
            
        except Exception as e:
            state_dict = state.model_dump()
            state_dict['error'] = f"Domain refinement failed: {str(e)}\nLLM Response: {response.generations[0][0].text if 'response' in locals() else 'No response'}"
            return WorkflowState(**state_dict).model_dump()

    workflow.add_node("analyze", analyze_domains)
    workflow.add_node("refine", refine_domains)
    
    workflow.add_edge("analyze", "refine")
    workflow.add_edge("refine", END)
    
    workflow.set_entry_point("analyze")
    return workflow.compile()

async def identify_domains(text: str) -> Dict[str, Any]:
    try:
        workflow = create_workflow()
        
        initial_state = WorkflowState(text=text).model_dump()
        final_state = await workflow.ainvoke(initial_state)
        # Sort domains by confidence
        domains = sorted(
            final_state["domains"],
            key=lambda x: x["confidence"],
            reverse=True
        )
    
        return {
            "status": "success",
            "domains": [domain for domain in domains],
            "metadata": {
                "total_domains": len(domains),
                "average_confidence": sum(d["confidence"] for d in domains) / len(domains) if domains else 0
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
    The patient presented with elevated blood pressure (150/90 mmHg) and reported 
    chest pain radiating to the left arm. ECG showed ST-segment elevation in leads 
    V1-V4. Initial troponin levels were elevated at 0.5 ng/mL. Treatment with 
    aspirin and nitroglycerin was initiated immediately.
    """
    
    result = asyncio.run(identify_domains(test_text))
    print(json.dumps(result, indent=2))