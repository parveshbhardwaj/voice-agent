from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LiveKit Settings
    LIVEKIT_HOST: str
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str

    # Inference Settings
    INFERENCE_API_ENDPOINT: str
    INFERENCE_API_KEY: str

    # Model Settings
    # MODEL_TEMPERATURE: float = 0.7
    # MAX_TOKENS: int = 800
    # TOP_P: float = 0.95
    # FREQUENCY_PENALTY: float = 0.0
    # PRESENCE_PENALTY: float = 0.0
    # STOP_SEQUENCE: Optional[str] = None
    
    # Agent Settings
    AGENT_INSTRUCTIONS: str = "You are a helpful voice AI assistant."

    class Config:
        env_file = ".env"