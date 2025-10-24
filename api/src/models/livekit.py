from pydantic import BaseModel,Field
from typing import List

class CreateRoomRequest(BaseModel):
    room_name: str
    user_id: str
    agent_name: str

class RoomResponse(BaseModel):
    room: str
    user_token: str
    agent_name: str
    agent_dispatched: bool

class RoomStatusResponse(BaseModel):
    room_name: str
    room_exists: bool
    participants: List[str]  = Field(default_factory=list)
    
    