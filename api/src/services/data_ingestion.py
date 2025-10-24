import os
import logging
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.xlsx import partition_xlsx
from genai.ingestion_pipeline_async import DataIngestionPipeline
from typing import Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DataIngestionService:

    def __init__(self):
        # self.embed_model = getAzureLLMIndexEmbeddingModel()
        # self.llm_model = getAzureLLMIndexModel()
        self.ingestion_pipeline = DataIngestionPipeline()
        self.ingestion_statuses: Dict[str, str] = {}  # Move status tracking to service
        
    async def process_data_and_ingest(
            self, 
            user_id: str, 
            project_id: str,
            project_name: str,
            doc_loc: str, 
            docs: list[str],
            submission_id: str
        ) -> Tuple[bool, str]:
        try:
            logger.info(f"Processing documents for user {user_id} at location {doc_loc}")
            self.ingestion_statuses[submission_id] = IngestionStatus.PROCESSING
            docs = [doc.strip() for doc in docs if doc.strip()]
            if not docs:
                self.ingestion_statuses[submission_id] = IngestionStatus.FAILED
                return False, "No valid documents provided for ingestion."
            # Your processing logic here
            llamaindex_ingestion_pipeline = await self.ingestion_pipeline.create_pipeline_with_llamindex(add_processes=True)
            documents = await self.ingestion_pipeline.load_documents_with_llamaindex(input_dir=doc_loc, input_files=docs)

            # TODO -- remove this comment - For testing purposes, I am just using 3 documents
            success,nodes = await self.ingestion_pipeline.process_documents(project_id=project_id,
                                                                            project_name=project_name,documents=documents, pipeline=llamaindex_ingestion_pipeline)
            
            if not success:
                self.ingestion_statuses[submission_id] = IngestionStatus.FAILED
                return False, "Failed to process documents."
            
            if not nodes:
                self.ingestion_statuses[submission_id] = IngestionStatus.FAILED
                return False, "No nodes generated from the documents."          
            
            store_success = await self.ingestion_pipeline.store_documents(user_id=user_id,nodes=nodes) 
            # , overwrite=False)
            print(f"store_success: {store_success}")

            if not store_success:
                self.ingestion_statuses[submission_id] = IngestionStatus.FAILED
                return False, "Failed to store processed documents."  

            self.ingestion_statuses[submission_id] = IngestionStatus.COMPLETED
            #TODO - remove after testing --- to test the fetch from the store
            # await self.ingestion_pipeline.get_query_engine(user_id,project_id=project_id,project_name=project_name) 
            
            logger.info(f"Documents processed and stored successfully for user {user_id}.")           
            return True, "Documents processed and ingested successfully."   
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            self.ingestion_statuses[submission_id] = IngestionStatus.FAILED
            return False, str(e)
        
    def get_ingestion_status(self, submission_id: str) -> str:
        return self.ingestion_statuses.get(submission_id, IngestionStatus.FAILED)
