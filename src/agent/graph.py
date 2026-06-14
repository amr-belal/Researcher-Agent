from langgraph.graph import StateGraph  , END , START
from .state import AgentState
from .nodes import think_node, act_node, reflect_node, writer_node , research_tools , observe_node





def create_graph():
    """
    Creates the state graph for the agent, defining the nodes and transitions based on the agent's behavior and interactions.
    This method should be implemented to define the specific states and transitions relevant to the agent's tasks and goals.
    The graph should include nodes representing different states of the agent, such as receiving a question, processing information, making decisions, and providing answers.   
    """

    graph = StateGraph(AgentState)

    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("write", writer_node)
    graph.add_node("observe", observe_node)


    graph.add_edge(START, "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", "observe")
    graph.add_edge("observe", "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda state: "write" if state["is_complete"] else "think"
    )
   
    
    graph.add_edge("write", END)


    return graph.compile()