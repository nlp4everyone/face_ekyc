from fastapi import HTTPException, status

class MediafileUnsupportedException(HTTPException):
    def __init__(self,
                 file_type :str = "image"):
        detail = f"Uploaded files must be under {file_type} format!"
        super().__init__(status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                         detail = detail)