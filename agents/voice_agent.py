import logging
from utils import getAzureSTTModel,getAzureLLMModel,getAzureTTSModel
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("voice-agent")

class VoiceAssistant(Agent):
    def __init__(self) -> None:
        # This project is configured to use Deepgram STT, OpenAI LLM and Cartesia TTS plugins
        # Other great providers exist like Cerebras, ElevenLabs, Groq, Play.ht, Rime, and more
        # Learn more and pick the best one for your app:
        # https://docs.livekit.io/agents/plugins

        super().__init__(
            instructions="You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
            "You were created as a demo to showcase the capabilities of LiveKit's agents framework.",
            stt=getAzureSTTModel(),
            llm=getAzureLLMModel(),
            tts=getAzureTTSModel(),
            # use LiveKit's transformer-based turn detector
            turn_detection=MultilingualModel(),
        )
    async def on_transcription(self, transcription, *args, **kwargs):
        # transcription.participant.identity or transcription.participant_id is usually available
        participant_id = getattr(transcription, "participant_id", None)
        if participant_id is None and hasattr(transcription, "participant"):
            participant_id = getattr(transcription.participant, "identity", None)
        print(f"Received transcription from participant: {participant_id}")
        print(f"Transcription text: {transcription.text if hasattr(transcription, 'text') else transcription}")

        # Optionally, you can use participant_id to customize your reply logic
        # For example, you could store a mapping of participant_id to session, etc.

        return await super().on_transcription(transcription, *args, **kwargs)

    async def on_enter(self):
        # The agent should be polite and greet the user when it joins :)
        self.session.generate_reply(
            instructions="Hey, how can I help you today?", allow_interruptions=True
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    # Log metrics and collect usage data
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
    )

    # Trigger the on_metrics_collected function when metrics are collected
    session.on("metrics_collected", on_metrics_collected)

    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(),
        room_input_options=RoomInputOptions(
            # enable background voice & noise cancellation, powered by Krisp
            # included at no additional cost with LiveKit Cloud
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="voice-assistant-agent",
        ),
    )