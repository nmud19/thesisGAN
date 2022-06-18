import pytorch_lightning as pl
from app.model import lit_model
from app.discriminator import patch_gan
from app.generator import unetGen
from app.consume_data import consume_data
import os


# TODO fix this unit test to work in pipeline
def test_lightning_model():
    """ Test model e2e """
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
    train_dataset, valid_dataset = consume_data.get_dataset(
        train_images=train_images,
        test_images=train_images,
    )
    train_dataloader, valid_dataloader = consume_data.get_dataloaders(
        train_dataset=train_dataset,
        test_dataset=train_dataset,
    )

    # Trainer
    epoch_inference_callback = lit_model.EpochInference(valid_dataloader)
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint()
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="pix2pix_lightning_model")
    trainer = pl.Trainer(
        fast_dev_run=True,
        # max_epochs=5,
        logger=logger,
        callbacks=[
            epoch_inference_callback,
            checkpoint_callback,
            # pl.callbacks.TQDMProgressBar(refresh_rate=10)
        ],
        default_root_dir="/Users/nimud/PycharmProject/thesisGAN/checkpoints/"
    )
    trainer.fit(
        model,
        train_dataloader,
        valid_dataloader
    )
