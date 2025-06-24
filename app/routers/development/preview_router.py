# FastAPI Component
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
# Image component
from app.utils.image import ImagePreprocess
from app.utils.face.alignment import BasicAlignment
from app.utils.face.embedding import calculate_similarity
from app.utils.face.frontal_metrics import FrontalFaceFiltering
# Video component
from app.utils.video import VideoReader
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
    # Get file type
    file_type = file.content_type
    if not (file_type.startswith("image/") or file_type.startswith("video/")):
        raise MediafileUnsupportedException(file_type = "image/video")
    # Get model
    face_detector = get_face_recognition_model()
    # Check model
    if face_detector is None:
        raise FaceDetectorUnavailableException()

    # Load file
    file_content = await file.read()

    if file_type.startswith("video"):
        # Get image from frame
        images_numpy = VideoReader.read_video_from_bytes(video_bytes = file_content)
        # Get detections
        detections = face_detector.detect_faces(images_numpy,
                                                batch_size = 16,
                                                max_size = 320)
    else:
        # Images case
        images_numpy = [ImagePreprocess.bytes_to_numpy(file_content)]
        # Get detections
        detections = face_detector.detect_faces(images_numpy,
                                                batch_size = 1,
                                                max_size = 320)
    # When face existed
    if len(detections) == 0:
        SystemLogger.warning("Cannot detect face from image")
        raise FaceNotFoundException()

    # Video case
    if len(detections) != 1:
        # Get selected image based in head pose
        selected_image, properties = FrontalFaceFiltering.select_frames(frames = images_numpy,
                                                                     detections = detections)
        # Normalize image
        final_image = BasicAlignment.align_face_5points(image = selected_image[0],
                                                        landmarks = properties[0][1].keypoint.model_dump())
    else:
        # Normalize image
        final_image = BasicAlignment.align_face_5points(image = images_numpy[0],
                                                        landmarks = detections[0].keypoints)
    # Encode image as JPEG
    _, encoded_image = cv2.imencode('.jpg', final_image)
    # Show up to API
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
        raise MediafileUnsupportedException()

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
    if not all(files_content): raise MediafileUnsupportedException()

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
                "matching": True if similarities.tolist()[0] > threshold else False,
                "embedding_model": face_embedding_model.model_name}

    except Exception as e:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT,
                            detail = e)