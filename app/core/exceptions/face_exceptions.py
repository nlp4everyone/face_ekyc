from fastapi import HTTPException, status

class FaceNotFoundException(HTTPException):
    def __init__(self):
        detail = "Cannot detect face from image!"
        super().__init__(status_code = status.HTTP_404_NOT_FOUND,
                         detail = detail)