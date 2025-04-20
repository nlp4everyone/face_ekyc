from fastapi import HTTPException, status

class UserNotFoundException(HTTPException):
    def __init__(self, user_id: str):
        detail = f"User with ID {user_id} not found"
        super().__init__(status_code = status.HTTP_404_NOT_FOUND,
                         detail = detail)

class UserExistedException(HTTPException):
    def __init__(self, user_id: str):
        detail = f"User with ID {user_id} has existed"
        super().__init__(status_code = status.HTTP_403_FORBIDDEN,
                         detail = detail)