"""Step 7: image transforms.

Images are already Phase 0 aligned/cropped -- no face detection, alignment,
or cropping happens here, only resize/augment/normalise. ITA and the
demographic labels never pass through here; they travel solely through
FaceDataset's metadata dict, never the image tensor.

ImageNet mean/std normalisation is used as a conventional default (the
backbone trains from scratch here, not fine-tuned from an ImageNet
checkpoint, so this is a choice, not a requirement -- flagged for review).
"""
from __future__ import annotations

import torchvision.transforms as T

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
