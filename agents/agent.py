from dotenv import load_dotenv
# import ssl
# import certifi
# import httpx
# import requests
from utils import getAzureSTTModel,getAzureLLMModel,getAzureTTSModel
from urllib3.exceptions import InsecureRequestWarning
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")
    
    async def on_transcription(self, transcription, *args, **kwargs):
        print(f"Received transcription: {transcription}")
        return await super().on_transcription(transcription, *args, **kwargs)

    async def on_reply(self, reply, *args, **kwargs):
        print(f"Agent reply: {reply}")
        return await super().on_reply(reply, *args, **kwargs)


async def entrypoint(ctx: agents.JobContext):
    print("into the entry point function...")

    session = AgentSession(
        # stt=openai.STT(model="gpt-4o-transcribe",client=client),
        # llm=openai.LLM(model="gpt-4o-mini",client=client),
        # tts = openai.TTS(model="gpt-4o-mini-tts",voice="ash",client=client,instructions="Speak in a friendly and conversational tone."),
        stt=getAzureSTTModel(),
        llm=getAzureLLMModel(),
        tts=getAzureTTSModel(),
        # openai.TTS(model="gpt-4o-mini-tts",voice="ash",instructions="Speak in a friendly and conversational tone."),

        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
                # LiveKit Cloud enhanced noise cancellation
                # - If self-hosting, omit this parameter
                # - For telephony applications, use `BVCTelephony` for best results
                # noise_cancellation=noise_cancellation.BVC(), 
            ),
            # ssl=ssl_context 
        )

    await ctx.connect()
    print(f"Room name: {ctx.room}")
    print(f"Agent name: {ctx.agent}")
    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )
    

if __name__ == "__main__":
    print("Starting the agent...")
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint,agent_name="assistant-agent"))