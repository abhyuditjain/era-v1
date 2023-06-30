from albumentations import (
    Compose,
    Normalize,
    HorizontalFlip,
    CoarseDropout,
    ShiftScaleRotate,
)
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


class Transforms:
    def __init__(self, means, stds, train=True):
        if train:
            self.transformations = Compose(
                [
                    HorizontalFlip(),
                    ShiftScaleRotate(),
                    CoarseDropout(
                        min_holes=1,
                        max_holes=1,
                        min_height=16,
                        max_height=16,
                        min_width=16,
                        max_width=16,
                        fill_value=[x * 255 for x in means],  # type: ignore
                        mask_fill_value=None,
                    ),
                    Normalize(mean=means, std=stds),
                    ToTensorV2(),
                ]
            )
        else:
            self.transformations = Compose(
                [
                    Normalize(mean=means, std=stds),
                    ToTensorV2(),
                ]
            )

    def __call__(self, img):
        return self.transformations(image=np.array(img))["image"]
