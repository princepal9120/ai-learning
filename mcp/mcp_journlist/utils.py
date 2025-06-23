from urllib.pars import quote_plus
from dotenv import load_dotenv
import os
from fastapi import FastAPI,HTTPException
import requests
from bs4 import BeautifulSoup
import ollama
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from elevenlabs import ElevenLabs

load_dotenv()


class MCPOverloadedError(Exception):
    """ "Custome Exception for MCP servie overloads"""

    pass


def generate_valid_news_url(keyword: str) -> str:
    """ "
    Genearate a Google New Search URL for a keyword with option sorting by latest

    Args:
        keyword: Search tern to use in the news search

    Returns:
        str: Constructed Google News search URL
    """
    q = quote_plus(keyword)
    return f"https://news.google.com/search?q={q}&tbs=sbd:1"


def generate_news_urls_to_scrape(list_of_keywords):
    valid_urls_dict = {}
    for keyword in list_of_keywords:
        valid_urls_dict[keyword] = generate_valid_news_url(keyword)

    return valid_urls_dict


def scrape_with_brightdata(url: str) -> str:
    """ "Scrape a URL using BrightData"""

    headers = {
        "Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_TOKEN')}",
        "Content-Type": "application/json",
    }

    payload = {
        "zone": os.getenv("BRIGHTDATA_WEB_UNLOKCER_ZONE"),
        "url": url,
        "format": "raw",
    }

    try:
        response = requests.post("https://api.brightdata.com/request", json=payload, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BrightData error: {str(e)}")
    


def clean_html_to_text(html_content: str)-> str:
    """Clean HTML content to plain text"""
    soup=BeautifulSoup(html_content, "html.parser")    
    text=soup.get_text(separator="\n")
    return text.strip()


def extract_headlines(cleaned_text: str)-> str:
    """
    Extract and concatentat headlines from cleaned news text content.
    
    Args:
        cleaned_text: Raw tex from news page after HTML cleaning
    
    Returns:
    str: combined headlines seprated by newlines
    """
    
    headlines =[]
    current_block=[]
    
    lines=[line.strip() for line in cleaned_text.split('\n') if line.strip()]
    
    for line in lines:
        if line=="MOre":
            if current_block:
                #First line of block is headline
                headlines.append(current_block[0])
                
                current_block=[]
            
        else: 
            current_block.append(line)
     
    if current_block:
        headlines.append(current_block[0])
        
        
    return "\n".join(headlines)        