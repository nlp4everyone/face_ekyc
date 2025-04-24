from fastapi import FastAPI
# Define route
from .routers.development import (basic_ekyc_router,
                                  preview_router,
                                  advanced_ekyc_router)
# Define startup
from .startup import (init_models,
                      init_minio_storage,
                      init_elastic_search)
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
    }
]
# Define app
app = FastAPI(openapi_tags = tags_metadata)
# Add preview router
app.include_router(preview_router,
                   prefix = "/development",
                   tags = [tags_metadata[0].get("name")])
# Add basic ekyc router
app.include_router(basic_ekyc_router,
                   prefix = "/development",
                   tags = [tags_metadata[0].get("name")])
# Add advanced ekyc router
app.include_router(advanced_ekyc_router,
                   prefix = "/development",
                   tags = [tags_metadata[0].get("name")])

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
    # Init Elastic Search
    es_service = init_elastic_search()
    await es_service.create_index()

    # Measure time for processing
    SystemLogger.success(f"Start up done after: {round(time.perf_counter() - start,1)}s")
    # Logging
    # SystemLogger.info(f"Start up done after: {round(time.perf_counter() - start,1)}s")