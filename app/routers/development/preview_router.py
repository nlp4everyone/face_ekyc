# FastAPI Component
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
# Image component
from app.utils.image import ImagePreprocess
from app.utils.face.alignment import BasicAlignment
from app.utils.face.embedding import calculate_similarity
# Startup
from app.startup import get_face_recognition_model, get_face_embedding_model
# Config
from app.core.config.constant import *
# Exception
from app.core.exceptions import *
# Logger
from loggers import SystemLogger
# Components
from io import BytesIO
from typing import List
from datetime import datetime
import cv2

# Define route
preview_router = APIRouter()

@preview_router.post("/face_align")
async def face_align(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise ImageTypeException()
    # Get model
    face_detector = get_face_recognition_model()
    # Check model
    if face_detector is None:
        raise FaceDetectorUnavailableException()

    # Load file
    file_content = await file.read()
    images_numpy = ImagePreprocess.bytes_to_numpy(file_content)
    # Get face
    detections = face_detector.detect_faces([images_numpy])
    # When face existed
    if len(detections) == 0:
        SystemLogger.warning("Cannot detect face from image")
        raise FaceNotFoundException()

    # Get detection
    detections = detections[0]
    landmarks = detections.keypoints
    # Align image
    aligned_image = BasicAlignment.align_face_5points(image = images_numpy,
                                                      landmarks = landmarks)
    # Encode image as JPEG
    _, encoded_image = cv2.imencode('.jpg', aligned_image)
    return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/jpeg")

@preview_router.post("/face_compare")
async def face_compare(files: List[UploadFile] = File(...,
                                                      description = "Upload images for comparison. First image is source, the rest is reference",
                                                      media_type = "image/png")):
    # Get model
    face_detector = get_face_recognition_model()
    face_embedding_model = get_face_embedding_model()
    # Check model
    if face_detector is None: raise FaceDetectorUnavailableException()
    if face_embedding_model is None: raise FaceEmbeddingUnavailableException()

    # Raise exception if not enough file
    if len(files) < 2: raise HTTPException(status_code = 400,
                                           detail = "Please provide at least 2 files")
    # Check input file, requires images at all
    files_content = [True if file.content_type.startswith("image/") else False for file in files]
    # Check file
    if not all(files_content):
        raise ImageTypeException()

    # Define start time
    begin_time = datetime.now().strftime("%Y/%d/%m %H:%M:%S")


    try:
        # Read as bytes
        images_byte = [await file.read() for file in files]
        # Convert as numpy
        images_numpy = [ImagePreprocess.bytes_to_numpy(image) for image in images_byte]

        # Detecting face
        face_detections = face_detector.detect_faces(images_numpy)

        # Get landmarks
        faces_landmark = [detection.keypoints for detection in face_detections]

        # Get faces aligned
        faces_aligned = [BasicAlignment.align_face_5points(image = image,
                                                           landmarks = landmark)for (landmark, image) in zip(faces_landmark,images_numpy)]
        # Embedding
        faces_embeddings = face_embedding_model.embed(faces_aligned)

        # Source embedding
        source_embedding, reference_embeddings = faces_embeddings[0], faces_embeddings[1:]
        # Calculate similarity
        similarities = calculate_similarity(source_embedding, reference_embeddings)
        return {"created_at": begin_time,
                "similarities": similarities.tolist(),
                "embedding_model": face_embedding_model.model_name}

    except Exception as e:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT,
                            detail = e)

@preview_router.post("/face_matching")
async def face_matching(files: List[UploadFile] = File(...,
                                                       description = "Upload images for comparison. First image is source, the rest is reference",
                                                       media_type = "image/png"),
                        threshold :float = DEFAULT_MATCHING_THRESHOLD):
    # Get model
    face_detector = get_face_recognition_model()
    face_embedding_model = get_face_embedding_model()
    # Check model
    if face_detector is None: raise FaceDetectorUnavailableException()
    if face_embedding_model is None: raise FaceEmbeddingUnavailableException()

    # Raise exception if not enough file
    if len(files) != 2: raise HTTPException(status_code = 400,
                                            detail = "Please provide only 2 image files")
    # Check input file, requires images at all
    files_content = [True if file.content_type.startswith("image/") else False for file in files]
    # Check type
    if not all(files_content): raise ImageTypeException()

    # Define start time
    begin_time = datetime.now().strftime("%Y/%d/%m %H:%M:%S")
    # Read as bytes
    images_byte = [await file.read() for file in files]

    try:
        # Convert as numpy
        images_numpy = [ImagePreprocess.bytes_to_numpy(image) for image in images_byte]
        # Detecting face
        face_detections = face_detector.detect_faces(images_numpy)

        # Get landmarks
        faces_landmark = [detection[0].keypoints for detection in face_detections]
        # Get faces aligned
        faces_aligned = [BasicAlignment.align_face_5points(image = image,
                                                           landmarks = landmark)for (landmark, image) in zip(faces_landmark,images_numpy)]
        # Embedding
        faces_embeddings = face_embedding_model.embed(faces_aligned)
        # Source embedding
        source_embedding, reference_embeddings = faces_embeddings[0], faces_embeddings[1:]
        # Calculate similarity
        similarities = calculate_similarity(source_embedding, reference_embeddings)
        return {"created_at": begin_time,
                "matching": True if similarities.tolist()[0] > threshold else False,
                "embedding_model": face_embedding_model.model_name}

    except Exception as e:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT,
                            detail = e)