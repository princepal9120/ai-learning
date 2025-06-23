from urllib.pars import quote_plus
from dotenv import load_dotenv
import os
from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
import ollama
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from elevenlabs import ElevenLabs

load_dotenv()


class MCPOverloadedError(Exception):
    """"Custome Exception for MCP servie overloads"""
    pass



def generate_valid_news_url(keyword: str) -> str:
    """"
    Genearate a Google New Search URL for a keyword with option sorting by latest
    
    Args:
        keyword: Search tern to use in the news search
        
    Returns:
        str: Constructed Google News search URL
    """
    q=quote_plus(keyword)
    return f"https://news.google.com/search?q={q}&tbs=sbd:1"


def generate_news_urls_to_scrape(list_of_keywords):
    valid_urls_dict={}
    for keyword in list_of_keywords:
        valid_urls_dict[keyword]=generate_valid_news_url(keyword)
        
    return valid_urls_dict


def scrape_with_birghtdata(url:str)-> str:
    """"Scrape a URL using BrightData"""
    
    headers = {
        "Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_TOKEN')}",
        "Content-Type": "application/json",
    }