from fastapi import HTTPException, status

class ImageNotFoundException(HTTPException):
    def __init__(self, image_name: str):
        detail = f"Image name: {image_name} not found in Minio"
        super().__init__(status_code = status.HTTP_404_NOT_FOUND,
                         detail = detail)