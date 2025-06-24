from fastapi import HTTPException, status

class ImageTypeException(HTTPException):
    def __init__(self):
        detail = "Uploaded files must be under image format!"
        super().__init__(status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                         detail = detail)