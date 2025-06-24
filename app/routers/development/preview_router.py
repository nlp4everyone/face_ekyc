# FastAPI Component
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
# Image component
from app.utils.image import ImagePreprocess
from app.utils.face.alignment import BasicAlignment
# Other components
from io import BytesIO
# Startup
from app.startup import get_face_recognition_model
# Exception
from app.core.exceptions import *
# Logger
from loggers import SystemLogger
# Components
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