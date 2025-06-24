from pydantic import BaseModel
from typing import List, Optional, Union
import numpy as np
class FaceDetection(BaseModel):
    box :Optional[List[int]] = None
    confidence :Optional[float] = None
    keypoints :Optional[dict] = None

class FacialKeyPoints(BaseModel):
    left_eye :Optional[List[float]] = None
    right_eye:Optional[List[float]] = None
    nose: Optional[List[float]] = None
    left_mouth: Optional[List[float]] = None
    right_mouth: Optional[List[float]] = None
    chin: Optional[List[float]] = None

class BaseDetector:
    def __init__(self):
        pass

    def detect_faces(self,
                     image: Union[str, np.ndarray],
                     limit: int = 1) -> List[FaceDetection]:
        raise NotImplementedError()