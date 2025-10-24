from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import MetadataMode
from llama_index.core import Settings
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import silero

from utils import (
    getAzureSTTModel,
    getAzureLLMModel,
    getAzureTTSModel,
    getAzureLLMIndexModel, 
    getAzureLLMIndexEmbeddingModel
)

# # check if storage already exists
# THIS_DIR = Path(__file__).parent
# PERSIST_DIR = THIS_DIR / "retrieval-engine-storage"
# if not PERSIST_DIR.exists():
#     # load the documents and create the index
#     documents = SimpleDirectoryReader(THIS_DIR / "./../data").load_data()
#     Settings.embed_model = getAzureLLMIndexEmbeddingModel()
#     Settings.llm = getAzureLLMIndexModel()
#     print(f"Loading the document ...")
#     index = VectorStoreIndex.from_documents(documents)
#     print(f"creating the index ...")
#     # store it for later
#     index.storage_context.persist(persist_dir=PERSIST_DIR)
#     print(f"persisted the index ...")
# else:
#     # load the existing index
#     Settings.embed_model = getAzureLLMIndexEmbeddingModel()
#     Settings.llm = getAzureLLMIndexModel()
#     print(f"Loading the existing index ...")
#     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#     print(f"creating the index ...")
#     index = load_index_from_storage(storage_context)

def load_index_from_db_storage(participant_id: str):
    """Get query engine for stored documents."""
    try:
        collection_name = "embedding_store"
        collection_name = f"{collection_name}_{participant_id}"
        chroma_client = chromadb.PersistentClient(path='./chroma_db')
        # chroma_client = chromadb.Client(chromadb.config.Settings(
        #                                 chroma_db_impl="duckdb+parquet",
        #                                 persist_directory='./chroma_db'
        #                         ))
        collection = chroma_client.get_collection(collection_name)

        # Set up vector store
        vector_store = ChromaVectorStore(chroma_collection=collection)            
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=getAzureLLMIndexEmbeddingModel()
        )
        return index
        
    except Exception as e:
        print(f"Error creating query engine: {e}")
        raise

class ReportRetrievalAgent(Agent):
    def __init__(self, index: VectorStoreIndex):
        super().__init__(
            instructions=(
                "You are a voice assistant. Your interface "
                "with users will be voice. You should use short and concise "
                "responses, and avoiding usage of unpronouncable punctuation." 
            ),
            vad=silero.VAD.load(),
            stt=getAzureSTTModel(),
            llm=getAzureLLMModel(),
            tts=getAzureTTSModel(),
        )
        self.index = index
        # self.job_ctx = job_ctx
    
    # async def on_enter(self):
    #     # The agent should be polite and greet the user when it joins :)
    #     print(f"local participant:{self.job_ctx.room.local_participant.identity}")

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ):
        user_msg = chat_ctx.items[-1]
        # print(f"RetrievalAgent: received user message: {user_msg}")
        assert isinstance(user_msg, llm.ChatMessage) and user_msg.role == "user"
        user_query = user_msg.text_content
        assert user_query is not None

        # Add table formatting instruction to the query
        # enhanced_query = f"""
        # {user_query}
        # If the answer contains numerical data, statistics, or structured information, 
        # present it in a simple table format using | as column separators. 
        # Keep table headers concise and clear.
        # Explain the table briefly before presenting it.
        # """
        enhanced_query=user_query
        retriever = self.index.as_retriever()
        print(f"RetrievalAgent: user query: {enhanced_query}")
        nodes = await retriever.aretrieve(enhanced_query)

        # print(f"RetrievalAgent: retrieved {len(nodes)} nodes for query: {enhanced_query}")
        instructions = ("Context that might help answer the user's question. ")
        for node in nodes:
            node_content = node.get_content(metadata_mode=MetadataMode.LLM)
            instructions += f"\n\n{node_content}"

        # update the instructions for this turn, you may use some different methods
        # to inject the context into the chat_ctx that fits the LLM you are using
        system_msg = chat_ctx.items[0]
        # print(f"RetrievalAgent: system message: {system_msg}")
        if isinstance(system_msg, llm.ChatMessage) and system_msg.role == "system":
            # TODO(long): provide an api to update the instructions of chat_ctx
            system_msg.content.append(instructions)
        else:
            chat_ctx.items.insert(0, llm.ChatMessage(role="system", content=[instructions]))
            safe_instructions = instructions.replace("\n", "\\n")
            # print(f"update instructions: {safe_instructions}...")

        # update the instructions for agent
        # await self.update_instructions(instructions)

        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    
    # Now you can access participant details
    print(f"Connected to room {ctx.room.name}")
    print(f"fetch infromation for participant {participant}")

    # Access the agent's participant object
    agent_participant = ctx.room.local_participant
    
    # Retrieve the metadata string
    metadata_str = agent_participant.metadata

    print(f"agent metadata",metadata_str)
    print(f"participant metadata",participant.metadata)
    #participatn metadata {"projectId":"proj_1749872575361_fbdgx94ql","projectName":"FinancialDocumentProject2"}
    # project_name = participant.metadata.get("projectName", "your project")
    # greeting = f" Hey, Looks like you need some help with {project_name}, how can I help you today?"
    greeting = "Hey, how can I help you today?"
    index = load_index_from_db_storage(participant_id=participant.identity)
    agent = ReportRetrievalAgent(index)
    
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

    await session.say(greeting, allow_interruptions=True)
    
  
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,agent_name="report-retrieval-agent"))