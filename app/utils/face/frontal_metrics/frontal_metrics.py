# Typing
from typing import Union
# Base point
from app.utils.face.detector import FaceDetection
from .base_metrics import FrontalProperties
# Other component
import numpy as np
import cv2

# Sample facial point
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
])

class MediapipeMetric:
    @staticmethod
    def _get_head_pose(image_points,
                       image_width,
                       image_height):
        """Estimate head point over image points"""
        focal_length = image_width
        center = (image_width / 2, image_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = euler_angles.flatten()
        return pitch, yaw, roll

    @staticmethod
    def calculate_frontalness_score(image_points,
                                    frame :np.ndarray) -> Union[float,None]:
        """
        Calculate score to measure how frontal of a face is.
        :param frame: Input image which has a face ( Numpy Array).
        :return: score (float): Higher is better (more frontal)
        """
        # Get the face information
        h, w = frame.shape[:2]
        # Calculate pitch, yaw, roll value
        pitch, yaw, roll = MediapipeMetric._get_head_pose(image_points,
                                                          image_width = w,
                                                          image_height = h)
        # *** Add more strategy like accumulate and weighted metrics
        return -(abs(pitch) + abs(yaw))

    @staticmethod
    def extract_facial_properties(detection :FaceDetection,
                                  frame :np.ndarray):
        """
        Calculate score to measure how frontal of a face is.
        :param frame: Input image which has a face ( Numpy Array).
        :return: score (float): Higher is better (more frontal)
        """
        h, w = frame.shape[:2]
        image_points = detection.keypoints
        image_points = np.array([tuple(image_points.get("nose")),
                                 tuple(image_points.get("chin")),
                                 tuple(image_points.get("right_eye")),
                                 tuple(image_points.get("left_eye")),
                                 tuple(image_points.get("right_mouth")),
                                 tuple(image_points.get("left_mouth"))
        ], dtype="double")

        #Calculate pitch, yaw, roll value
        pitch, yaw, roll = MediapipeMetric._get_head_pose(image_points,
                                                          image_width = w,
                                                          image_height = h)

        # Return
        return FrontalProperties(score = -(abs(pitch) + abs(yaw)),
                                 pitch = pitch,
                                 yaw = yaw,
                                 roll = roll,
                                 keypoint = detection.keypoints,
                                 image_height = h,
                                 image_width = w)

