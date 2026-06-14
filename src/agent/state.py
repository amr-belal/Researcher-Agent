
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    Represents the state of the agent, including its current task, progress, and any relevant information.
    """
    question: str
    
    messages: Annotated[list[BaseMessage], operator.add, "The most recent message from the agent."]

    # information retireved by tools 
    research_findings: list[dict] # [ {tool: info}, ... ]

    final_answer: str

    is_complete: bool
    