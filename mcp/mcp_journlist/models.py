from pydantic import BaseModel
from typing import List

class NewsRequest(BaseModel):
    topic: List[str]
    source_type: str