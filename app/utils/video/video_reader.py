# Component
import tempfile, cv2
import numpy as np
# Typing
from typing import List


class VideoReader:
    @staticmethod
    def read_video_from_bytes(video_bytes: bytes,
                              skip_interval :int = 3) -> List[np.ndarray]:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            temp_video.write(video_bytes)
            temp_video.flush()

            cap = cv2.VideoCapture(temp_video.name)
            frames = []
            count = 0

            # Open Camera
            while cap.isOpened():
                count += 1
                if count % skip_interval == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)  # Each `frame` is a NumPy array (H x W x 3)
            cap.release()

        return frames