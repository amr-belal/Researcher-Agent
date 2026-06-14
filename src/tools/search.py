from langchain_core.tools import tool
# from duckduckgo_search import DDGS
from ddgs import DDGS



@tool
def search_tool(query:str) -> list[dict]:
    """ 
    Search the web for information about a given query.
    Returns a list of results with title, url, and snippet.
    """
    
    with DDGS() as ddgs:

        results = ddgs.text(query, max_results=5)
    
    return [
        {
            "title": r["title"],
            "url": r["href"],
            "snippet": r["body"],
            
        }
        for r in results
    ]