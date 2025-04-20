import numpy as np
import cv2, torch
from torchvision import transforms

class ImagePreprocess:
    @staticmethod
    def bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Use cv2.IMREAD_GRAYSCALE for grayscale
        return img

    @staticmethod
    def convert_numpy_to_torch(bgr_image :np.ndarray) ->torch.Tensor:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # Convert the image to a PyTorch tensor
        transform = transforms.ToTensor()
        return transform(rgb_image)