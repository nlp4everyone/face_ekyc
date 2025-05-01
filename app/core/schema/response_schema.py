from pydantic import BaseModel
from typing import Optional
from strenum import StrEnum

class RequestResult(BaseModel):
    request_id: str
    status :str
    request_type :str
    data :Optional[dict] = None
    created_at :Optional[str] = None
    updated_at :Optional[str] = None

class RetrievedResult(BaseModel):
    id :Optional[str] = None
    score :float
    payload :dict

class ResponseStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class RequestType(StrEnum):
    QUERY = "QUERY"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"