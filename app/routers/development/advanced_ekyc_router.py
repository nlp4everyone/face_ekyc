# FastAPI Component
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Form
# Image component
from app.utils.image import ImagePreprocess
from app.utils.face.alignment import BasicAlignment
# Startup
from app.startup import (get_face_recognition_model,
                         get_face_embedding_model,
                         get_minio_storage,
                         get_qdrant_service)
# Other components
from datetime import datetime
# Image model
from app.utils.face.embedding import AdaFaceEmbedding, TimmEmbedding
from app.utils.face.recognition import MTCNNRecognition
# Minio
from app.db.minio import MinioObjectStorage
from app.db.qdrant import QdrantService, ApiException
# Log
from loggers import SystemLogger
# Schema
from app.core.schema import FaceRequest

# ekyc router
advanced_ekyc_router = APIRouter()


@advanced_ekyc_router.post("/face_register")
async def face_register(face_id :str = Form(...),
                        face_name :str = Form(...),
                        file: UploadFile = File(...)):
    # Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                            detail = "Uploaded files must be under image format!")
    # Get model
    mtcnn : MTCNNRecognition = get_face_recognition_model()
    face_embedding_model : TimmEmbedding = get_face_embedding_model()
    # Minio
    minio_storage :MinioObjectStorage = get_minio_storage()
    # Qdrant
    qdrant_service = get_qdrant_service()

    # Define start time
    begin_time = datetime.now().strftime("%Y/%d/%m %H:%M:%S")
    # Read as bytes
    image_byte = await file.read()
    # Convert as numpy
    image_numpy = ImagePreprocess.bytes_to_numpy(image_byte)

    # Detecting face
    face_detection = mtcnn.detect_faces(image_numpy)
    if len(face_detection) == 0:
        raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                            detail = "No found face!")

    try:
        SystemLogger.info("Trying add new face to database: ...")
        # Get landmarks
        face_landmark = face_detection[0].keypoints
        # Get faces aligned
        face_aligned = BasicAlignment.align_face_5points(image = image_numpy, landmarks = face_landmark)
        # Embedding
        face_embedding = face_embedding_model.embed(face_aligned)

        # Convert embedding to list of float
        face_vector = face_embedding.squeeze(0).tolist()

        # Validate
        face_information = FaceRequest(embedding = face_vector,
                                       face_id = face_id,
                                       face_name = face_name)
        # Append to Qdrant
        inserted_result = await qdrant_service.insert_face_embedding(face_information)
        # Check status
        if inserted_result.status.COMPLETED == "completed":
            SystemLogger.info("Add new face vector to Qdrant")

        # Upload image to minio
        minio_storage.upload_image(image = image_numpy, image_name = file.filename)
        # Return
        return {"status": "completed",
                "face_id": face_id,
                "face_name": face_name,
                "url": ""}

    except ApiException as e:
        SystemLogger.error(f"Face id: {face_id} has existed!")
        raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                            detail = "Face has existed! Cannot insert.")
