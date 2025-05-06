# Depedencies
import timm, torch, cv2
import torch.nn.functional as F
# Typing
from typing import Literal, Union, List
# Other dependencies
import numpy as np
from torchvision import transforms

TIMM_MODEL = Literal["hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3",
"hf_hub:gaunernst/vit_tiny_patch8_112.cosface_ms1mv3",
"hf_hub:gaunernst/vit_tiny_patch8_112.arcface_ms1mv3",
"hf_hub:gaunernst/vit_tiny_patch8_112.adaface_ms1mv3"]
class Utils:
    @staticmethod
    def convert_numpy_to_torch(bgr_image: np.ndarray) -> torch.Tensor:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # Convert the image to a PyTorch tensor
        transform = transforms.ToTensor()
        return transform(rgb_image)

class TimmEmbedding:
    def __init__(self,
                 model_name :TIMM_MODEL = "hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3",
                 device :Literal["cpu","cuda"] = "cuda"):
        # Define device
        self._device = device
        self._model_name = model_name
        self._model = timm.create_model(model_name, pretrained=True).eval().to(self._device)

    @property
    def model_name(self):
        return self._model_name.split("/")[-1]

    @staticmethod
    def convert_numpy_to_torch(bgr_image: np.ndarray) -> torch.Tensor:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # Convert the image to a PyTorch tensor
        transform = transforms.ToTensor()
        return transform(rgb_image)

    def embed(self,
              images :Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        # Convert string or ndarray to List[string]
        if isinstance(images, str) or isinstance(images, np.ndarray): images = [images]
        # Check condition
        if len(images) == 0:
            raise Exception("Images cannot be empty!")

        # Get list of numpy image
        images_torch = [Utils.convert_numpy_to_torch(image) for image in images]
        # Stack image
        stack_images = torch.stack(images_torch).to(self._device)
        embs = self._model(stack_images)  # output shape (1, 512)
        return F.normalize(embs, dim=1)