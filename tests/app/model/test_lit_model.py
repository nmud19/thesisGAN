import pytorch_lightning as pl
from app.model import lit_model
from app.discriminator import patch_gan
from app.generator import unetGen
from app.consume_data import consume_data
import os


def test_lightning_model():
    """ Test model e2e """
    trainer = pl.Trainer(fast_dev_run=True)
    generator = unetGen.UNET()
    discriminator = patch_gan.PatchGan(
        input_channels=6,
        hidden_channels=1
    )
    model = lit_model.Pix2PixLitModule(
        generator=generator,
        discriminator=discriminator
    )
    train_dir_path = "/Users/nimud/PycharmProject/thesisGAN/sample_data"
    train_images = [
        f"{train_dir_path}/{x}" for x in os.listdir(train_dir_path)
    ]
    train_dataset, _ = consume_data.get_dataset(
        train_images=train_images,
        test_images=train_images,
    )
    train_dataloader, _ = consume_data.get_dataloaders(
        train_dataset=train_dataset,
        test_dataset=train_dataset,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
    )