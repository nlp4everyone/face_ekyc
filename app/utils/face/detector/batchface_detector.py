# Module
from batch_face import RetinaFace, LandmarkPredictor
# Typing
from typing import Literal, List, Union, Any
import numpy as np
# Inheritance
from .base_detector import BaseDetector, FaceDetection

class BatchFaceDetector(BaseDetector):
    def __init__(self,
                 gpu_id :int = 0,
                 model_path: Any = None,
                 network: str = "mobilenet",
                 landmark_backbone :str = "MobileNet",
                 device: str = "cuda",
                 return_dict: bool = False,
                 fp16: bool = False):
        super().__init__()
        # Params
        self._gpu_id = gpu_id
        self._model_path = model_path
        self._network = network
        self._device = device
        self._return_dict = return_dict
        self._fp16 = fp16
        self._backbone = landmark_backbone
        # Detector
        self._detector = RetinaFace(gpu_id = self._gpu_id,
                                    model_path = self._model_path,
                                    network = self._network,
                                    device = self._device,
                                    return_dict = self._return_dict,
                                    fp16 = self._fp16)
        # Predictor
        self._predictor = LandmarkPredictor(gpu_id = self._gpu_id,
                                            backbone = self._backbone)

    def detect_faces(self,
                     images :List[np.ndarray],
                     threshold :float = 0.6,
                     batch_size :int = 2,
                     max_size :int = 512) -> List[FaceDetection]:
        """
        Detect face from images
        :param images: List of RBG images
        :param threshold:
        :param preprocess:
        :param batch_size:
        :param max_size:
        :return:
        """
        # Get detections
        detection_results = self._detector.pseudo_batch_detect(images = images,
                                                               threshold = threshold,
                                                               max_size = max_size,
                                                               batch_size = batch_size)
        # When empty face
        if len(detection_results[0]) == 0:
            return []

        outputs = []
        # Interate each image detection
        for detection_result in detection_results:
            # In case frame only has 1 bounding box
            if len(detection_result) == 1:
                # Append it to list
                outputs.append(detection_result)
                continue

            # List face bounding box from images ( Multiple face)
            bboxes = [bbox.tolist() for (bbox,_,_) in detection_result]
            # Compute area for each box
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]

            # Get index of the bounding box has largest area
            largest_area_index = areas.index(max(areas))
            # Append to list
            outputs.append([detection_result[largest_area_index]])

        # Get the landmark
        landmarks = self._predictor(outputs, images, from_fd=True)
        # Return face detections
        return self._convert_to_face_detection(outputs, landmarks)

    @staticmethod
    def _get_facial_landmark(landmarks :np.ndarray):
        chin = landmarks[8]  # Point of the chin
        nose_tip = landmarks[30]  # Tip of the nose
        left_eye = (landmarks[36] + landmarks[39]) / 2
        right_eye = (landmarks[42] + landmarks[45]) / 2
        left_mouth = landmarks[48]  # Left corner of mouth
        right_mouth = landmarks[54]
        return {"chin": chin.tolist(),
                "nose": nose_tip.tolist(),
                "left_eye": left_eye.tolist(),
                "right_eye": right_eye.tolist(),
                "left_mouth": left_mouth.tolist(),
                "right_mouth": right_mouth.tolist()}

    @staticmethod
    def _convert_to_face_detection(inputs :List[List[tuple]],
                                   landmarks :List[List[np.ndarray]]):
        results = []
        for (input,landmark) in zip(inputs, landmarks):
            # Get bounding box, keypoint and confident
            box, _, conf = input[0]
            # Get
            landmark = BatchFaceDetector._get_facial_landmark(landmark[0])
            # Append to list ( Box is rounded to most nearest integer)
            results.append(FaceDetection(box = np.round(box).astype(int).tolist(),
                                         confidence = round(float(conf),ndigits = 3),
                                         keypoints = landmark))
        return results