from fastapi import HTTPException, status

class FaceDetectorUnavailableException(HTTPException):
    def __init__(self):
        detail = "Face Detector Service Unavailable!"
        super().__init__(status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
                         detail = detail)

class FaceEmbeddingUnavailableException(HTTPException):
    def __init__(self):
        detail = "Face Embedding Service Unavailable!"
        super().__init__(status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
                         detail = detail)