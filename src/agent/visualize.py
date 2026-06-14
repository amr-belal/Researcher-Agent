from src.agent.graph import create_graph
graph = create_graph()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")