import os

import pytest
import torch
from pytorch_lightning import Trainer

from helpers.data_creation import MNISTDataModule
from helpers.simpleGAN.GAN import GAN

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

def test_simple_gan_works()->None : 
    try : 
        dm = MNISTDataModule()
        model = GAN(*dm.size())
        trainer = Trainer(
            gpus=AVAIL_GPUS, 
            max_epochs=5, 
            progress_bar_refresh_rate=20,
            fast_dev_run=True
        )
        trainer.fit(model, dm)
    except Exception as e : 
        pytest.fail(f"Simple GAN exception - {e}")
