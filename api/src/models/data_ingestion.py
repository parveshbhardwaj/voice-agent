from pydantic import BaseModel,Field
from typing import List

class DataIngestionRequest(BaseModel):
    user_id: str
    project_id: str
    project_name: str
    doc_location: str
    docs:List[str]

class DataIngestionResponse(BaseModel):
    message: str
    submission_id: str

    
    