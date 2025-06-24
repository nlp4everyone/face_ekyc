from pydantic import BaseModel
from typing import Optional, Tuple
# Base point
from app.utils.face.detector import FacialKeyPoints

class FrontalProperties(BaseModel):
    score :Optional[float] = None
    pitch :Optional[float] = None
    yaw :Optional[float] = None
    roll :Optional[float] = None
    keypoint :Optional[FacialKeyPoints] = None
    bbox :Optional[Tuple[int,int,int,int]] = None
    image_height :Optional[int] = None
    image_width :Optional[int] = None