from .state import AgentState
from .llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from src.tools.search import search_tool
from src.tools.wikipedia import wikipedia_tool
from src.tools.arxiv import arxiv_search




llm = get_llm()

research_tools = [search_tool, wikipedia_tool, arxiv_search]

act_node = ToolNode(tools=research_tools)

llm_with_tools = llm.bind_tools(research_tools)


def think_node(state: AgentState) -> dict:
    """
    This node represents the agent's thinking process, where it processes the received question, retrieves relevant information, and formulates a response.
    The node should take the current state of the agent, including the question and any retrieved information, and use the LLM to generate a response or make decisions based on the input.
    The output of this node should update the agent's state with any new information or insights gained from the thinking process.
    """

    question = state["question"]
    messages = state["messages"]
    research_findings = state["research_findings"]

    system = SystemMessage(content=f"""
    You are a research agent. Your job is to answer the user's question by using tools.
    Think step by step: what do you need to search for next?
    If you have enough information, say FINAL ANSWER.

    Question: {question}
    Research so far: {research_findings}
    """)

    response  =  llm_with_tools.invoke([system] + messages)
    
    return {"messages": [response]}



def reflect_node(state: AgentState) -> dict:
    """
    This node represents the agent's reflection process, where it evaluates the information gathered from the research and the response generated in the thinking node.
    The node should take the current state of the agent, including the question, research findings, and the response from the thinking node, and use the LLM to reflect on whether the response is sufficient or if further research is needed.
    The output of this node should update the agent's state with any new insights or decisions made during the reflection process.
    """

    question = state["question"]
    messages = state["messages"]
    research_findings = state["research_findings"]

    if not state["research_findings"]:
        return {"is_complete": False}
    
    
    system = SystemMessage(content=f"""
            You are evaluating if we have enough information to answer the question.
            Reply with ONLY one word: "complete" or "incomplete"

            Question: {question}
            Research so far: {research_findings}
            Last response: {messages[-1].content}
            """)

    response  =  llm.invoke([system] + messages)
    content = response.content.strip()
    is_done = content == "complete" or content.startswith("complete")
    
    return {"messages": [response],"is_complete": is_done }



def writer_node(state: AgentState)->dict:
    """
    This node represents the agent's writing process, where it takes the information gathered from the research and the response generated in the thinking node to formulate a final answer to the user's question.
    The node should take the current state of the agent, including the question, research findings, and any insights from the reflection node, and use the LLM to generate a coherent and comprehensive answer to the user's question.
    The output of this node should be the final answer that can be presented to the user.
    """
    
    question = state["question"]
    messages = state["messages"]
    research_findings = state["research_findings"]

    system = SystemMessage(content=f"""
                You are a research writer. Based on the research findings, write a clear and comprehensive answer to the user's question.
                Be concise, accurate, and cite your sources.

                Question: {question}
                Research findings: {research_findings}
                """)

    response  =  llm.invoke([system] + messages)
    
    return {"final_answer": response.content}


def observe_node(state: AgentState) -> dict:
    """
    This node represents the agent's observation process, where it takes the current state of the agent, including the question, research findings, and any responses generated in the previous nodes, and updates the agent's state with any new insights or information gained from the observation.
    The node should use the LLM to analyze the current state and extract any relevant information or insights that can be used to further refine the agent's understanding of the question and guide its future actions.
    The output of this node should update the agent's state with any new observations or insights gained from this process.
    """

    question = state["question"]
    # messages = state["messages"]
    last_message =state["messages"][-1] if state["messages"] else None
    research_findings = state["research_findings"]

    
    
    return {
        "research_findings": state["research_findings"] + [{"content": last_message.content}]
    }