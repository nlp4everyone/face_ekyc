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
from app.db.qdrant import QdrantService
# Log
from loggers import SystemLogger
# Schema
from app.core.schema import FaceRequest
# Exception
from app.core.exceptions import (UserNotFoundException,
                                 UserExistedException,
                                 FaceNotFoundException)
from app.core.config.constant import DEFAULT_SIMILARITY_TOP_K

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
    # Resized to fixed size image (Reducing time for processing)
    image_numpy = ImagePreprocess.resize_image_keep_aspect_ratio(image_numpy,
                                                                 fixed_width = 512)

    try:
        # Detecting face
        face_detection = mtcnn.detect_faces(image_numpy)

        # Logging
        SystemLogger.info("Trying add new face to database: ...")
        # Get landmarks
        face_landmark = face_detection[0].keypoints
        # Get faces aligned
        face_aligned = BasicAlignment.align_face_5points(image = image_numpy,
                                                         landmarks = face_landmark)
        # Embedding
        face_embedding = face_embedding_model.embed(face_aligned)

        # Convert embedding to list of float
        face_vector = face_embedding.squeeze(0).tolist()

        # Validate
        face_information = FaceRequest(embedding = face_vector,
                                       face_id = face_id,
                                       face_name = face_name,
                                       image_name = file.filename)
        # Append to Qdrant
        inserted_result = await qdrant_service.insert_face_embedding(face_information)
        # Check status
        if inserted_result.status == "FAILED":
            # Logging
            SystemLogger.error("Failed to add face to Qdrant")
            raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                                detail = "Failed to add face to Qdrant")

        # Logging
        SystemLogger.success("Add new face vector to Qdrant")
        # Upload image to minio (With compressed version of image)
        compressed_image = ImagePreprocess.compress_image(image_numpy,
                                                          quality = 70)
        # Upload image
        result = await minio_storage.aupload_image(image = compressed_image,
                                                   image_name = file.filename)
        # Return
        return inserted_result
    except UserExistedException as e:
        SystemLogger.error(f"Face id: {face_id} has existed!")
        raise UserExistedException(user_id = face_id)
    except FaceNotFoundException as e:
        SystemLogger.error(f"Cannot found face")
        raise FaceNotFoundException()

@advanced_ekyc_router.delete("/face_delete")
async def face_delete(face_id :str):
    # Minio
    minio_storage :MinioObjectStorage = get_minio_storage()
    # Qdrant
    qdrant_service :QdrantService = get_qdrant_service()

    try:
        # Delete object from Qdrant
        response = await qdrant_service.delete_point(face_id = face_id)
        SystemLogger.success(f"Remove face {face_id} from Qdrant")

        # Remove object from Minio (If existed)
        await minio_storage.aremove_image(response.data.get("image_name"))
        # Logging
        SystemLogger.success(f"Remove face {face_id} from Minio")
        # Return
        return response
    except UserNotFoundException as e:
        SystemLogger.error(f"Face id: {face_id} not found!")
        raise UserNotFoundException(user_id = face_id)

@advanced_ekyc_router.post("/face_retrieve")
async def face_retrieve(file: UploadFile = File(...),
                        similarity_top_k :int = Form(DEFAULT_SIMILARITY_TOP_K)):
    # Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                            detail = "Uploaded files must be under image format!")
    # Qdrant
    qdrant_service :QdrantService = get_qdrant_service()
    # Get model
    mtcnn: MTCNNRecognition = get_face_recognition_model()
    face_embedding_model: TimmEmbedding = get_face_embedding_model()

    # Read as bytes
    image_byte = await file.read()
    # Convert as numpy
    image_numpy = ImagePreprocess.bytes_to_numpy(image_byte)
    # Resized to fixed size image (Reducing time for processing)
    image_numpy = ImagePreprocess.resize_image_keep_aspect_ratio(image_numpy,
                                                                 fixed_width = 512)

    # Detecting face
    face_detection = mtcnn.detect_faces(image_numpy)
    if len(face_detection) == 0:
        raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                            detail = "No found face!")

    # Get landmarks
    face_landmark = face_detection[0].keypoints
    # Get faces aligned
    face_aligned = BasicAlignment.align_face_5points(image = image_numpy,
                                                     landmarks = face_landmark)
    # Embedding
    face_embedding = face_embedding_model.embed(face_aligned)
    # Convert embedding to list of float
    face_vector = face_embedding.squeeze(0).tolist()
    # Logging
    SystemLogger.success(f"Trying to create retrieved request: ...")
    # Retrieve point
    retrieve_points = await qdrant_service.retrieve_points(embedding = face_vector,
                                                           similarity_top_k = similarity_top_k)
    return {"points": retrieve_points}