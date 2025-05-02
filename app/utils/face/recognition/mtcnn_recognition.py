from facenet_pytorch import MTCNN
# Base Recognition
from .base_recognition import BaseRecognition, FaceDetection
# Other component
from typing import Literal, Union, List
import numpy as np
from app.core.exceptions import FaceNotFoundException

class MTCNNRecognition(BaseRecognition):
    def __init__(self,
                 device :Union[Literal["cpu","cuda:0"],str] = "cpu"):
        super().__init__()
        # Define MTCNN
        self._detector = MTCNN(thresholds = [0.7, 0.7, 0.8],
                               keep_all = True,
                               device = device,
                               select_largest = False)

    def detect_faces(self,
                     image: Union[str, np.ndarray],
                     limit: int = 1) -> List[FaceDetection]:
        # Detect
        boxes, probs, landmarks = self._detector.detect(image, landmarks = True)
        # Raise exceptions while not found face
        if boxes is None or probs is None:
            raise FaceNotFoundException()

        predictions = []
        # Iterate each
        for (box, prob, landmark) in zip(boxes, probs, landmarks):
            # Convert ndarray to list of int
            box = [int(element) for element in box.tolist()]
            landmark = [[int(x), int(y)] for x, y in landmark.tolist()]

            # Append values
            predictions.append(FaceDetection(box = box,
                                             confidence = prob,
                                             keypoints = {"left_eye": landmark[0],
                                                          "right_eye": landmark[1],
                                                          "nose": landmark[2],
                                                          "mouth_left": landmark[3],
                                                          "mouth_right": landmark[4]}))
        return predictions

    def batch_detect_faces(self,
                           images: List[Union[str, np.ndarray]]) -> List[List[FaceDetection]]:
        if isinstance(images, np.ndarray): images = [images]
        # Return
        return [self.detect_faces(image) for image in images]