from pydantic import BaseModel
from typing import List

class FaceRequest(BaseModel):
    embedding :List[float]
    face_id :str
    face_name :str
    image_name :str