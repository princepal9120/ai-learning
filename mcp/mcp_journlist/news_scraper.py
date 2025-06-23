import asyncio
import os
from typing import Dict, List
from aiolimiter import AsynceLimiter
from tenacity import retry,retry_if_exception_type, stop_after_attempt, wait_exponential
from langgraph.prebuild import create_react_agent
from dotenv import load_dotenv

from utils import generate_news_urls_to_scrape, scrape_with_birghtdata

from mcp import ClientSession, StdioServerlParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()


class NewsScraper:
    _rate_limiter= AsynceLimiter(5,1)
    @retry(
        stop=stop_after_attempt()
        wait=wait_exponential(multiplier=1,min=2,max=10)
    )
    
    async def scrape_news(self, topics: List[str])->Dict[str,str]:
        """Scrape and analyze news articles"""
        results={}
        
        for topic in topics:
            async with self._rate_limiter:
                try:
                    urls=generate_news_urls_to_scrape([topic])
                    search_html=scrape_with_birghtdata(urls[topic])
                    clean_text=
