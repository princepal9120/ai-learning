from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool
from crewai.tools import tool

from langchain_community.tools import DuckDuckGoSearchResults



@tool
def search_web_tool(query: str):
    """
    Search the web and return results
    """

    search_tool=DuckDuckGoSearchResults(num_results=10, verbose=True)

    return search_tool.run(query)