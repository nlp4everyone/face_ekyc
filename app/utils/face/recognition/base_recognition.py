from pydantic import BaseModel
from typing import List, Optional, Union
import numpy as np
class FaceDetection(BaseModel):
    box :Optional[List[int]] = None
    confidence :Optional[float] = None
    keypoints :Optional[dict] = None

class BaseRecognition:
    def __init__(self):
        pass

    def detect_faces(self,
                     image: Union[str, np.ndarray],
                     limit: int = 1) -> List[FaceDetection]:
        raise NotImplementedError()