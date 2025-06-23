from models import NewsRequests
from fastapi import FastApi
form 


app =FastApi()

@app.post("/generate-news-audio")
async def generate_news_audio(requests: NewsRequests):
    try:
        results={}
        
        #Scrape data

        if requests.source_type in ["news" "both"]:
            #scrape news
            results["news"] = {"news_scrapped": "this is from google newss"}
        
        if requests.source_type in ["reddit","both"]:
            #scrape reddit
            results["reddit"] ={"reddit_scrapped": "This is from reddit"}
        
        news_data= results.get("news", {})
        reddit_data=results.get("reddit",{})

        #setup LLM summarizer

        news_summary=my_summary_function(news_data, reddit_data)

        audio_path=convert_text_to_audio(news_summary)
    except Exception as e:
        return f"Unexcepted error {str(e)}"