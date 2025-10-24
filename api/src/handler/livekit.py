from fastapi import APIRouter, HTTPException
from models.livekit import CreateRoomRequest, RoomResponse , RoomStatusResponse
from services.livekit import LiveKitService

router = APIRouter()
livekit_service = LiveKitService()

@router.post("/create-room", response_model=RoomResponse)
async def create_room_and_dispatch_agent(req: CreateRoomRequest) -> RoomResponse:

    # Dispatch agent
    dispatch_success,token = await livekit_service.create_token_with_agent_dispatch(req.room_name, req.agent_name, req.user_id)
    if not dispatch_success:
        raise HTTPException(status_code=500, detail="Failed to dispatch agent")

    return RoomResponse(
        room=req.room_name,
        user_token=token,
        agent_name=req.agent_name,
        agent_dispatched=dispatch_success
    )

@router.get("/check/{room_name}", response_model=RoomStatusResponse)
async def check_room_and_agent_status(room_name: str) -> RoomStatusResponse:
    """
    Check if a room exists and if an agent is registered in it.
    """
    try:
        room_exists, participants = await livekit_service.check_room_participants(room_name)
        print(" control reached here 3")
        return RoomStatusResponse(
            room_name=room_name,
            room_exists=room_exists,
            participants=participants
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check room and agent status: {str(e)}")