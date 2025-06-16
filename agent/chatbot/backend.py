from pydantic import BaseModal
from typing import List
from fastapi import FastAPI


class RequestState(BaseModal):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool


app=FastAPI()    


