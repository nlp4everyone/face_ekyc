from fastapi import FastAPI
# Define route
from .routers.development import preview_router
from .routers import ekyc_router
# Define startup
from .startup import (init_models,
                      init_minio_storage,
                      init_qdrant_service)
# Define middle ware
from .core.middle_ware import TimerMiddleware
# Components
import time
from loggers import SystemLogger

# Tags
tags_metadata = [
    {
        "name": "Development",
        "description": "Contain features such as face alignment, face comparison for development step",
    },
    {
        "name": "Ekyc",
        "description": "Contain features for ekyc-related function for using in production",
    }
]
# Define app
app = FastAPI(openapi_tags = tags_metadata)
# Add preview router
app.include_router(preview_router,
                   prefix = "/development",
                   tags = [tags_metadata[0].get("name")])
# Add advanced ekyc router
app.include_router(ekyc_router,
                   prefix = "/api",
                   tags = [tags_metadata[1].get("name")])

# Add middleware
app.add_middleware(TimerMiddleware)

@app.on_event("startup")
async def startup_event():
    # Start
    start = time.perf_counter()
    # Init ml model
    init_models()
    # Init Minio
    init_minio_storage()
    # Init qdrant
    qdrant_service = init_qdrant_service()
    await qdrant_service.create_collection()

    # Measure time for processing
    SystemLogger.success(f"Start up done after: {round(time.perf_counter() - start,1)}s")
    # Logging
    # SystemLogger.info(f"Start up done after: {round(time.perf_counter() - start,1)}s")