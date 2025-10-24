from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
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

# check if storage already exists
THIS_DIR = Path(__file__).parent
PERSIST_DIR = THIS_DIR / "retrieval-engine-storage"
if not PERSIST_DIR.exists():
    # load the documents and create the index
    documents = SimpleDirectoryReader(THIS_DIR / "./../data").load_data()
    Settings.embed_model = getAzureLLMIndexEmbeddingModel()
    Settings.llm = getAzureLLMIndexModel()
    print(f"Loading the document ...")
    index = VectorStoreIndex.from_documents(documents)
    print(f"creating the index ...")
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"persisted the index ...")
else:
    # load the existing index
    Settings.embed_model = getAzureLLMIndexEmbeddingModel()
    Settings.llm = getAzureLLMIndexModel()
    print(f"Loading the existing index ...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    print(f"creating the index ...")
    index = load_index_from_storage(storage_context)

class RetrievalAgent(Agent):
    def __init__(self, index: VectorStoreIndex):
        super().__init__(
            instructions=(
                "You are a voice assistant created by LiveKit. Your interface "
                "with users will be voice. You should use short and concise "
                "responses, and avoiding usage of unpronouncable punctuation."
            ),
            vad=silero.VAD.load(),
            stt=getAzureSTTModel(),
            llm=getAzureLLMModel(),
            tts=getAzureTTSModel(),
        )
        self.index = index

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ):
        user_msg = chat_ctx.items[-1]
        print(f"RetrievalAgent: received user message: {user_msg}")
        assert isinstance(user_msg, llm.ChatMessage) and user_msg.role == "user"
        user_query = user_msg.text_content
        assert user_query is not None

        retriever = self.index.as_retriever()
        print(f"RetrievalAgent: user query: {user_query}")
        nodes = await retriever.aretrieve(user_query)

        print(f"RetrievalAgent: retrieved {len(nodes)} nodes for query: {user_query}")
        instructions = "Context that might help answer the user's question:"
        for node in nodes:
            node_content = node.get_content(metadata_mode=MetadataMode.LLM)
            instructions += f"\n\n{node_content}"

        # update the instructions for this turn, you may use some different methods
        # to inject the context into the chat_ctx that fits the LLM you are using
        system_msg = chat_ctx.items[0]
        print(f"RetrievalAgent: system message: {system_msg}")
        if isinstance(system_msg, llm.ChatMessage) and system_msg.role == "system":
            # TODO(long): provide an api to update the instructions of chat_ctx
            system_msg.content.append(instructions)
        else:
            chat_ctx.items.insert(0, llm.ChatMessage(role="system", content=[instructions]))
            safe_instructions = instructions.replace("\n", "\\n")
            print(f"update instructions: {safe_instructions}...")

        # update the instructions for agent
        # await self.update_instructions(instructions)

        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = RetrievalAgent(index)
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

    await session.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,agent_name="retrieval-agent"))