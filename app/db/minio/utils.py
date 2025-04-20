import numpy as np
import cv2
from io import BytesIO

def convert_image_to_bytes(image :np.ndarray):
    # Convert NumPy array to bytes (e.g., PNG format)
    _, encoded_image = cv2.imencode('.png', image)
    # Step 4: Prepare image bytes to upload
    return BytesIO(encoded_image.tobytes())
