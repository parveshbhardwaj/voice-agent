from config import settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from livekit.plugins import openai

def getAzureLLMIndexModel() -> AzureOpenAI:
    """
    Get the Azure OpenAI model for vector store indexing.
    """
    
    if not settings.INFERENCE_API_ENDPOINT or not settings.INFERENCE_API_KEY:
        raise ValueError("INFERENCE_API_ENDPOINT and INFERENCE_API_KEY must be set in environment variables.")
    
    azure_llm = AzureOpenAI(
        azure_deployment="gpt-4o-mini",
        azure_endpoint=settings.INFERENCE_API_ENDPOINT,
        api_key=settings.INFERENCE_API_KEY,
        api_version="2025-01-01-preview",
    )
    return azure_llm

def getAzureLLMIndexEmbeddingModel() -> AzureOpenAIEmbedding:
    """
    Get the Azure OpenAI embedding model for vector store indexing.
    """
    
    if not settings.INFERENCE_API_ENDPOINT or not settings.INFERENCE_API_KEY:
        raise ValueError("INFERENCE_API_ENDPOINT and INFERENCE_API_KEY must be set in environment variables.")
    
    embed_model = AzureOpenAIEmbedding(
        azure_deployment="text-embedding-3-small",
        azure_endpoint=settings.INFERENCE_API_ENDPOINT,
        api_key=settings.INFERENCE_API_KEY,
        api_version="2023-05-15",
    )
    return embed_model

def getAzureLLMModel() -> openai.LLM:
    """
    Get the Azure OpenAI embedding model for vector store indexing.
    """
    
    if not settings.INFERENCE_API_ENDPOINT or not settings.INFERENCE_API_KEY:
        raise ValueError("INFERENCE_API_ENDPOINT and INFERENCE_API_KEY must be set in environment variables.")

    azure_llm = openai.LLM.with_azure(
        azure_deployment="gpt-4o-mini",
        azure_endpoint=settings.INFERENCE_API_ENDPOINT,
        api_key=settings.INFERENCE_API_KEY,
        api_version="2024-04-01-preview"
    )
    return azure_llm

def getAzureSTTModel() -> openai.STT:
    """
    Get the Azure OpenAI STT model.
    """
    
    if not settings.INFERENCE_API_ENDPOINT or not settings.INFERENCE_API_KEY:
        raise ValueError("INFERENCE_API_ENDPOINT and INFERENCE_API_KEY must be set in environment variables.")
    
    azure_stt = openai.STT.with_azure(
        azure_deployment="gpt-4o-transcribe",
        azure_endpoint=settings.INFERENCE_API_ENDPOINT,
        api_key=settings.INFERENCE_API_KEY,
        api_version="2025-03-01-preview", 
    )
    return azure_stt

def getAzureTTSModel() -> openai.TTS:
    """
    Get the Azure OpenAI TTS model.
    """
    
    if not settings.INFERENCE_API_ENDPOINT or not settings.INFERENCE_API_KEY:
        raise ValueError("INFERENCE_API_ENDPOINT and INFERENCE_API_KEY must be set in environment variables.")
    
    azure_tts = openai.TTS.with_azure(
        azure_deployment="gpt-4o-mini-tts",
        azure_endpoint=settings.INFERENCE_API_ENDPOINT,
        api_key=settings.INFERENCE_API_KEY,
        api_version="2025-03-01-preview", 
    )
    return azure_tts