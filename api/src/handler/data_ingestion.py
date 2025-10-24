from fastapi import APIRouter, HTTPException
from models.data_ingestion import DataIngestionRequest, DataIngestionResponse 
from services.data_ingestion import DataIngestionService
import logging
import uuid
import asyncio
from enum import Enum
from typing import Dict
logger = logging.getLogger(__name__)

router = APIRouter()
data_ingestion_service = DataIngestionService()

# In-memory status tracking (use Redis/DB in production)
ingestion_statuses: Dict[str, str] = {}

@router.post("/data-ingest", response_model=DataIngestionResponse)
async def ingest_document(req: DataIngestionRequest) -> DataIngestionResponse:

    try:
        # Generate unique submission ID
        submission_id = str(uuid.uuid4())
        
        print(f"Received data ingestion request: {req}")
        # Start background task for processing
        asyncio.create_task(
            data_ingestion_service.process_data_and_ingest(
                user_id=req.user_id,
                project_id=req.project_id,
                project_name=req.project_name,
                doc_loc=req.doc_location,
                docs=req.docs,
                submission_id=submission_id
            )
        )
        return DataIngestionResponse(
            message="Documet ingestion started in background.",
            submission_id=submission_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start ingestion for documents: {str(e)}"
        )

@router.get("/status/{submission_id}")
async def get_status(submission_id: str):
    status = data_ingestion_service.get_ingestion_status(submission_id)
    return {"status": status}