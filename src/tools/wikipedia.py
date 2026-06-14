from langchain_core.tools import tool
import wikipediaapi
from ddgs import DDGS

@tool
def wikipedia_tool(query: str) -> list[dict]:
    """
    Search for information on Wikipedia.
    Use this for general knowledge questions or when you want a summary of a topic.
    Returns a list of results with title, url, and summary.
    """

    with DDGS() as ddgs:
        results = ddgs.text(f"site:wikipedia.org {query}", max_results=1)
    
    if not results:
        return []

    title = results[0]["title"].replace(" - Wikipedia", "")
    
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='ResearchAgent/1.0'
    )
    page = wiki.page(title)
    
    if page.exists():
        return [{"title": page.title, "url": page.fullurl, "summary": page.summary}]
    
    return []