from os import path, walk
from warnings import warn

import torch

from torch.utils.data import Dataset

from torchvision.io import decode_image

from torchvision.transforms.v2 import Transform, ToDtype
from torchvision.transforms.v2.functional import InterpolationMode

from PIL import Image

from model import PixelCrush


class ImageFolder(Dataset):
    ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})

    IMAGE_MODE = "RGB"

    def __init__(
        self,
        root_path: str,
        target_resolution: int,
        downscale_ratio: float,
        pre_transformer: Transform | None = None,
    ):
        if downscale_ratio not in PixelCrush.AVAILABLE_DOWNSCALE_RATIOS:
            raise ValueError(
                f"Upscale ratio must be either 2, 4, or 8, {downscale_ratio} given."
            )

        image_paths = []
        dropped = 0

        for folder_path, _, filenames in walk(root_path):
            for filename in filenames:
                if self.has_image_extension(filename):
                    image_path = path.join(folder_path, filename)

                    image = Image.open(image_path)

                    width, height = image.size

                    if width < target_resolution or height < target_resolution:
                        dropped += 1

                        continue

                    image_paths.append(image_path)

        if dropped > 0:
            warn(
                f"Dropped {dropped} images that were smaller "
                f"than the target resolution of {target_resolution}."
            )

        post_transformer = ToDtype(torch.float32, scale=True)

        self.image_paths = image_paths
        self.pre_transformer = pre_transformer
        self.post_transformer = post_transformer

    @classmethod
    def has_image_extension(cls, filename: str) -> bool:
        _, extension = path.splitext(filename)

        return extension in cls.ALLOWED_EXTENSIONS

    def __getitem__(self, index: int):
        hr_path, lr_path = self.image_paths[index]

        hr_image = decode_image(hr_path, mode=self.IMAGE_MODE)
        lr_image = decode_image(lr_path, mode=self.IMAGE_MODE)

        if self.pre_transformer:
            hr_image = self.pre_transformer(hr_image)
            lr_image = self.pre_transformer(lr_image)

        x = self.post_transformer(hr_image)
        y = self.post_transformer(lr_image)

        return x, y

    def __len__(self):
        return len(self.image_paths)
