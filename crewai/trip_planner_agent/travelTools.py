from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

# Web search tool using DuckDuckGo
@tool
def search_web_tool(query: str):
    """
    Searches the web and returns results.
    """
    search_tool = DuckDuckGoSearchResults(num_results=10, verbose=True)
    return search_tool.run(query)

# Web scraping tool
#web_search_tool = WebsiteSearchTool()
#scrape_website_tool = ScrapeWebsiteTool()