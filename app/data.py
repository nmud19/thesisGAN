import torch
import os
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import albumentations as A
import numpy as np
import albumentations.pytorch as al_pytorch
from typing import Dict, Tuple


class AnimeDataset(torch.utils.data.Dataset):
    """ Sketchs and Colored Image dataset """

    def __init__(self, imgs_path: List[str], transforms: transforms.Compose) -> None:
        """ Set the transforms and file path """
        self.list_files = imgs_path
        self.transform = transforms

    def __len__(self) -> int:
        """ Should return number of files """
        return len(self.list_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Get image and mask by index """
        # read image file
        img_path = img_file = self.list_files[index]
        image = np.array(Image.open(img_path))

        # divide image into sketchs and colored_imgs, right is sketch and left is colored images
        # as according to the dataset
        sketchs = image[:, image.shape[1] // 2:, :]
        colored_imgs = image[:, :image.shape[1] // 2, :]

        # data augmentation on both sketchs and colored_imgs
        augmentations = self.transform.both_transform(image=sketchs, image0=colored_imgs)
        sketchs, colored_imgs = augmentations['image'], augmentations['image0']

        # conduct data augmentation respectively
        sketchs = self.transform.transform_only_input(image=sketchs)['image']
        colored_imgs = self.transform.transform_only_mask(image=colored_imgs)['image']
        return sketchs, colored_imgs


class Transforms:
    """ Class to hold transforms """

    def __init__(self):
        # use on both sketchs and colored images
        self.both_transform = A.Compose([
            A.Resize(width=1024, height=1024),
            A.HorizontalFlip(p=.5)
        ],
            additional_targets={'image0': 'image'}
        )

        # use on sketchs only
        self.transform_only_input = A.Compose([
            # A.ColorJitter(p=.1),
            A.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5], max_pixel_value=255.0),
            al_pytorch.ToTensorV2(),
        ])

        # use on colored images
        self.transform_only_mask = A.Compose([
            A.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5], max_pixel_value=255.0),
            al_pytorch.ToTensorV2(),
        ])