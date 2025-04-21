import numpy as np
import cv2, torch
from torchvision import transforms
from io import BytesIO

class ImagePreprocess:
    @staticmethod
    def bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Use cv2.IMREAD_GRAYSCALE for grayscale
        return img

    @staticmethod
    def convert_image_to_bytes(image: np.ndarray):
        # Convert NumPy array to bytes (e.g., PNG format)
        _, encoded_image = cv2.imencode('.jpeg', image)
        # Step 4: Prepare image bytes to upload
        return BytesIO(encoded_image.tobytes())

    @staticmethod
    def convert_numpy_to_torch(bgr_image :np.ndarray) ->torch.Tensor:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # Convert the image to a PyTorch tensor
        transform = transforms.ToTensor()
        return transform(rgb_image)

    @staticmethod
    def resize_image_keep_aspect_ratio(image: np.ndarray,
                                       fixed_width: int = 512):
        # Check type
        if not isinstance(image, np.ndarray):
            raise ValueError("Image not found or cannot be opened.")
        # Get original dimensions
        original_height, original_width = image.shape[:2]
        # Return if image width less than fixed width
        if original_width <= fixed_width:
            return image

        # Calculate the new height to maintain the aspect ratio
        aspect_ratio = original_height / original_width
        new_height = int(fixed_width * aspect_ratio)

        # Resize the image
        return cv2.resize(image, (fixed_width, new_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def compress_image(image :np.ndarray,
                       quality :int = 70) -> BytesIO:
        # Compress using JPEG with given quality
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded_image = cv2.imencode('.jpg', image, encode_params)

        # When not success, return original image under BytesIO
        if not success:
            return ImagePreprocess.convert_image_to_bytes(image)

        # Convert to BytesIO
        return BytesIO(encoded_image.tobytes())