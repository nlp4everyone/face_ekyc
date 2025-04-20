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
from app.utils.face.embedding import AdaFaceEmbedding
from app.utils.face.recognition import MTCNNRecognition
# Minio
from app.db.minio import MinioObjectStorage

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
    ada_face : AdaFaceEmbedding = get_face_embedding_model()
    # Minio
    minio_storage :MinioObjectStorage = get_minio_storage()

    # Define start time
    begin_time = datetime.now().strftime("%Y/%d/%m %H:%M:%S")
    # Read as bytes
    images_byte = await file.read()
    # Convert as numpy
    images_numpy = ImagePreprocess.bytes_to_numpy(images_byte)

    # Detecting face
    face_detections = mtcnn.detect_faces(images_numpy)
    if len(face_detections) == 0:
        raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                            detail = "No found face!")

    try:
        # Get landmarks
        faces_landmark = [detection[0].get("keypoints") for detection in face_detections]
        # Get faces aligned
        faces_aligned = [BasicAlignment.align_face_5points(image = image,
                                                           landmarks = landmark)for (landmark, image) in zip(faces_landmark,images_numpy)]
        # Embedding
        faces_embeddings = ada_face.embed(faces_aligned)
        # Upload image to minio
        minio_storage.upload_image(image = images_numpy,
                                   image_name = file.filename)

    except Exception as e:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT,
                            detail = e)

