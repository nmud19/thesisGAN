import torch
import os
from typing import List, Optional
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import albumentations as A
import numpy as np
import albumentations.pytorch as al_pytorch
from typing import Dict, Tuple
from app import config
import pytorch_lightning as pl

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


class AnimeSketchDataModule(pl.LightningDataModule):
    """ Class to hold the Anime sketch Data"""

    def __init__(
            self,
            data_dir: str,
            num_images: int,
            train_folder_name: str = "train/",
            val_folder_name: str = "val/"
    ):
        super().__init__()
        self.data_dir: str = data_dir
        # Set train and val images folder
        train_path: str = f"{self.data_dir}{train_folder_name}/"
        train_images: List[str] = [f"{train_path}{x}" for x in os.listdir(train_path)]
        val_path: str = f"{self.data_dir}{val_folder_name}"
        val_images: List[str] = [f"{val_path}{x}" for x in os.listdir(val_path)]
        #
        self.train_images = train_images[:num_images] if num_images else train_images
        self.val_images = val_images[:num_images] if num_images else val_images

    def set_datasets(self) -> None:
        """ Get the train and test datasets """
        self.train_dataset = AnimeDataset(
            imgs_path=self.train_images,
            transforms=Transforms()
        )
        self.val_dataset = AnimeDataset(
            imgs_path=self.val_images,
            transforms=Transforms()
        )
        print("The train test dataset lengths are : ", len(self.train_dataset), len(self.val_dataset))
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        self.set_datasets()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.train_batch_size,
            shuffle=False,
            num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=4
        )
