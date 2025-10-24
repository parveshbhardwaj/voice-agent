from livekit import api,protocol
from config import settings
from livekit.rtc import Room
import logging
logger = logging.getLogger(__name__)

class LiveKitService:

    async def check_room_participants(self, room_name: str) -> tuple[bool, list[str]]:
        """
        Check if room exists and agent is registered.
        Returns: (room_exists: bool, agent_registered: bool)
        """
        try:
            lkapi = api.LiveKitAPI(
                settings.LIVEKIT_HOST,
                settings.LIVEKIT_API_KEY,
                settings.LIVEKIT_API_SECRET
            )

            # Check if room exists
            try:
                room_exists = False
                rooms_response = await lkapi.room.list_rooms(api.ListRoomsRequest(names=[room_name]))
                print("room_name_list",rooms_response)
                if rooms_response is None or not rooms_response.rooms:
                    logger.info(f"Room '{room_name}' does not exist.")
                else:
                    room_exists = True
            except Exception as e:
                logger.error(f"Error checking room: {e}")

            if room_exists:
                participants_response = await lkapi.room.list_participants(api.ListParticipantsRequest(room=room_name))
                participants_list = [p.identity for p in participants_response.participants]
   
            await lkapi.aclose()
            return room_exists, participants_list

        except Exception as e:
            logger.error(f"Error checking room and agent: {e}")
            await lkapi.aclose()
            return False, []
        
    async def create_token_with_agent_dispatch(self, room_name: str, agent_name: str, user_id: str):
        try:
            # Check if room and agent already exist
            # room_exists, agent_registered = await self.check_room_and_agent(room_name, agent_name)
            
            # # If agent is already registered, return success
            # if room_exists and agent_registered:
            #     logger.info(f"Agent '{agent_name}' already registered in room '{room_name}'")
            #     return True, self.generate_token(user_id, room_name)
            token = (
                api.AccessToken()
                .with_identity(user_id)
                .with_grants(api.VideoGrants(room_join=True, room=room_name))
                .with_room_config(
                    api.RoomConfiguration(
                        agents=[api.RoomAgentDispatch(agent_name=agent_name, metadata="agent-{user_id}".format(user_id=user_id))],
                    ),
                )
                .to_jwt()
            )
            roomJ = Room()
            await roomJ.connect(settings.LIVEKIT_HOST, token)
            return True, token
        except Exception as e:
            print(f"Error dispatching agent: {e}")
            return False, " "
        

    # async def dispatch_agent(self, room_name: str, agent_name: str, user_id: str) -> bool:
    #     try:
    #         service = api.AgentDispatchService(
    #             settings.LIVEKIT_HOST,
    #             settings.LIVEKIT_API_KEY,
    #             settings.LIVEKIT_API_SECRET
    #         )
    #         req = protocol.CreateDispatchRequest(
    #             room=room_name,
    #             agent_name=agent_name,
    #             metadata=f'{{"user_id": "{user_id}"}}'
    #         )
    #         await service.CreateDispatch(req)
    #         return True
    #     except Exception as e:
    #         print(f"Error dispatching agent: {e}")
    #         return False