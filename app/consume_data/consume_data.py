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
from app import config

torch.__version__


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
        resized_sketch = self.transform.resize_572(image=sketchs)
        resized_coloured_imgs = self.transform.resize_572(image=colored_imgs)

        # conduct data augmentation respectively
        sketchs = self.transform.transform_only_input(image=resized_sketch['image'])['image']
        colored_imgs = self.transform.transform_only_mask(image=resized_coloured_imgs['image'])['image']
        return sketchs, colored_imgs


class Transforms:
    """ Class to hold transforms """

    def __init__(self):
        # use on both sketchs and colored images
        self.resize_572 = A.Compose([
            A.Resize(width=572, height=572)
        ])

        self.resize_388 = A.Compose([
            A.Resize(width=388, height=388)
        ])

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


def get_dataset(train_images: str, test_images: str) -> Tuple[AnimeDataset, AnimeDataset]:
    """ Get the train and test datasets """
    train_dataset = AnimeDataset(
        imgs_path=train_images,
        transforms=Transforms()
    )
    test_dataset = AnimeDataset(
        imgs_path=test_images,
        transforms=Transforms()
    )
    print("The train test dataset lengths are : ", len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset


def get_dataloaders(train_dataset: AnimeDataset, test_dataset: AnimeDataset) -> Tuple[
    torch.utils.data.DataLoader, torch.utils.data.DataLoader,
]:
    """ Get DataLoaders pytorch """
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False
    )
    return train_dataloader, test_dataloader
