from langchain_core.tools import tool
import arxiv 

@tool 
def arxiv_search(query:str) -> list[dict]:
    """
    Search for scientific papers on arxiv.
    Use this for academic, scientific, or research questions.
    Returns a list of papers with title, url, and summary.
    """

    client = arxiv.Client()


    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance   
    )

    result = client.results(search)

    return [
        {
            "title": r.title,
            "url": r.entry_id,
            "summary": r.summary,
        }
        for r in result
    ]