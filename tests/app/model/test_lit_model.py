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
        discriminator=discriminator,
        use_gpu=False
    )
    # data Module
    anime_sketch_data_module = consume_data.AnimeSketchDataModule(
        data_dir="/Users/nimud/PycharmProject/thesisGAN/sample_data/",
        train_folder_name="",
        val_folder_name="",
        num_images=0
    )
    # Trainer
    # epoch_inference_callback = lit_model.EpochInference(valid_dataloader,use_gpu=False)
    # checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint()
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="lightning_logs")
    trainer = pl.Trainer(
        fast_dev_run=True,
        max_epochs=3,
        logger=logger,
        # callbacks=[
        #     # epoch_inference_callback,
        #     # checkpoint_callback,
        #     # pl.callbacks.TQDMProgressBar(refresh_rate=10)
        # ],
        default_root_dir="chk",
        progress_bar_refresh_rate=1
    )
    trainer.fit(
        model=model,
        datamodule=anime_sketch_data_module
    )
    print("complete!")
