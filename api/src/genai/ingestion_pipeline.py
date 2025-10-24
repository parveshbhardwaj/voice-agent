from typing import List, Optional, Tuple
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import chromadb
import logging
from llama_index.core.ingestion import IngestionPipeline
from utils import getAzureLLMIndexModel, getAzureLLMIndexEmbeddingModel
import os

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
    ):
        self.llm = getAzureLLMIndexModel()
        self.embed_model = getAzureLLMIndexEmbeddingModel()
        self.chroma_persist_dir = chroma_persist_dir
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
      
        # Initialize transformations
        self.text_splitter = SentenceSplitter(
            separator=" ",
            chunk_size=1024,
            chunk_overlap=128
        )
        self.summary_extractor = SummaryExtractor(llm=self.llm, nodes=5)
        self.qa_extractor = QuestionsAnsweredExtractor(llm=self.llm, questions=3)

    def create_pipeline_with_llamindex(self,add_processes: bool = True) -> IngestionPipeline:
        """Create the ingestion pipeline."""
        try:
        # Apply transformations pipeline
            pipeline_transformations = [self.text_splitter]
            
            if add_processes:
                pipeline_transformations.extend([
                    self.summary_extractor,
                    self.qa_extractor
                ])

            pipeline = IngestionPipeline(transformations=pipeline_transformations)
            
            return pipeline
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            return None
            
    def load_documents_with_llamaindex(
        self,
        input_dir: str,
        input_files: Optional[List[str]] = None
    ) -> List[Document]:
        """Load documents from directory."""
        try:
                   # Ensure input_dir exists
            if not os.path.exists(input_dir):
                raise ValueError(f"Directory does not exist: {input_dir}")

            # Validate files exist if specified
            files_list = list()
            if input_files:
                for file in input_files:
                    full_path = os.path.join(input_dir, file)
                    if not os.path.exists(full_path):
                        raise ValueError(f"File not found: {full_path}")
                    files_list.append(full_path)

           
            print(f"Loading documents from {input_dir} with files: {input_files}")
            reader = SimpleDirectoryReader(
                input_dir=input_dir,
                input_files=files_list,
                filename_as_id=True
            )
            return reader.load_data()
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def process_documents(
        self,
        documents: List[Document],
        pipeline: IngestionPipeline 
    ) -> Tuple[bool, List[Document]]:
        """Process and transform documents."""
        try:
            if not pipeline:
                logger.error("Pipeline is not initialized.")
                return False, []  
            if not documents:
                logger.error("No documents to process.")
                return False, []
            
            # Run the pipeline on the documents 
            nodes = pipeline.run(
                documents=documents,
                in_place=True,
                show_progress=True
            )
            return True, nodes
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return False, []

    def store_documents(
        self,
        user_id: str,
        nodes: List[Document],
        overwrite: bool = False
    ) -> bool:
        """Store documents in vector store."""
        collection_name = "embedding_store"
        collection_name = f"{collection_name}_{user_id}"
        print(f"Storing documents in collection: {collection_name}")

        try:
            if overwrite:
                self.chroma_client.delete_collection(collection_name)
                collection = self.chroma_client.create_collection(collection_name)
            else:
                collection = self.chroma_client.get_or_create_collection(name=collection_name)
            
            print(f"Collection created: {collection}")
            # print(f"Number of nodes to store: {len(nodes)}")
            # Set up vector store
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model
            )

            index.storage_context.persist(persist_dir=self.chroma_persist_dir)
            logger.info(f"Documents stored successfully in collection: {collection_name}")
            print(f"index persisted sucessfully...")
            return True
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            return False

    def get_query_engine(self, user_id: str, similarity_top_k: int = 3):
        """Get query engine for stored documents."""
        try:
            collection_name = "embedding_store"
            collection_name = f"{collection_name}_{user_id}"

            collection = self.chroma_client.get_collection(collection_name)

            # Set up vector store
            vector_store = ChromaVectorStore(chroma_collection=collection)
            # storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # index = load_index_from_storage(storage_context)

            

            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embed_model
            )

            # from llama_index.core.retrievers import VectorIndexRetriever
            # retriever = VectorIndexRetriever(index=index,embed_model=self.embed_model)
            query_engine = index.as_query_engine(
                llm=self.llm,
                similarity_top_k=similarity_top_k
            )

            # from llama_index.core.query_engine import RetrieverQueryEngine

            # query_engine = RetrieverQueryEngine(retriever)
            # response = query_engine.query("What are the main findings in the financial report?")
            # print(response)

            response = query_engine.query("list the name of board of directors")
            print(response)

            response = query_engine.query("show me the group organizational structure")
            print(response)
            
        except Exception as e:
            logger.error(f"Error creating query engine: {e}")
            raise