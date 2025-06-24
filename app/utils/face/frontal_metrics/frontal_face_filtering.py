from typing import List, Tuple
import numpy as np
from app.utils.face.frontal_metrics import MediapipeMetric
from app.utils.face.detector import FaceDetection
from app.utils.face.frontal_metrics import FrontalProperties

class FrontalFaceFiltering:
    @staticmethod
    def select_frames(frames :List[np.ndarray],
                      detections :List[FaceDetection],
                      top_k :int = 1) -> Tuple[List[np.ndarray],List[Tuple[int,FrontalProperties]]]:
        # Score for evaluating frontal
        frontal_scores = [MediapipeMetric.extract_facial_properties(detection = detection,
                                                                    frame = frame) for (detection,frame) in zip(detections,frames)]
        # Add index and remove None value
        indexed_scores = [(i,score) for (i, score) in enumerate(frontal_scores) if score is not None]
        # Sort value based in score descendingly:
        sorted_indexed_scores = sorted(indexed_scores,key = lambda x: x[1].score, reverse= True)

        # Select top-k element with highest score ( Most frontal)
        selected_scores = sorted_indexed_scores[:top_k]
        # Return frame with highest score
        return [frames[index] for (index, _) in selected_scores],selected_scores