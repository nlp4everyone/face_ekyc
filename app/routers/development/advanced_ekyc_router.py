# FastAPI Component
from fastapi import APIRouter, File, UploadFile, HTTPException, status
# Image component
from app.utils.image import ImagePreprocess
from app.utils.face.alignment import BasicAlignment
# Startup
from app.startup import (get_face_recognition_model,
                         get_face_embedding_model,
                         get_minio_storage)
# Other components
from datetime import datetime
# Image model
from app.utils.face.embedding import AdaFaceEmbedding, TimmEmbedding
from app.utils.face.recognition import MTCNNRecognition
# Minio
from app.db.minio import MinioObjectStorage
# Log
from loggers import SystemLogger

# ekyc router
advanced_ekyc_router = APIRouter()

@advanced_ekyc_router.post("/face_register")
async def face_register(file: UploadFile = File(...)):
    # Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                            detail = "Uploaded files must be under image format!")
    # Get model
    mtcnn : MTCNNRecognition = get_face_recognition_model()
    face_embedding_model : TimmEmbedding = get_face_embedding_model()
    # Minio
    minio_storage :MinioObjectStorage = get_minio_storage()

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
        # Get landmarks
        face_landmark = face_detection[0].keypoints
        # Get faces aligned
        face_aligned = BasicAlignment.align_face_5points(image = image_numpy, landmarks = face_landmark)
        # Embedding
        face_embeddings = face_embedding_model.embed(face_aligned)
        # Upload image to minio
        minio_storage.upload_image(image = image_numpy, image_name = file.filename)

    except Exception as e:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT,
                            detail = e)

