from fastapi import FastAPI
from handler.livekit import router as room_router
from handler.data_ingestion import router as data_ingestion_router

app = FastAPI(title="Voice Agent")
app.include_router(room_router, prefix="/api/v1/rooms")
app.include_router(data_ingestion_router, prefix="/api/v1/pipeline")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000,log_level="debug")