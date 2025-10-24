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