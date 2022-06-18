import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision


class Pix2PixLitModule(pl.LightningModule):
    """ Lightning Module for pix2pix """

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def __init__(self, generator, discriminator,lambda_recon=200):
        super().__init__()
        self.save_hyperparameters()

        self.gen = generator
        self.disc = discriminator

        # intializing weights
        self.gen = self.gen.apply(self._weights_init)
        self.disc = self.disc.apply(self._weights_init)
        #
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step(self, real_images, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(conditioned_images)
        disc_logits = self.disc(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)
        lambda_recon = self.hparams.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.gen(conditioned_images).detach()
        fake_logits = self.disc(fake_images, conditioned_images)

        real_logits = self.disc(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def forward(self, x):
        return self.gen(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, condition = batch

        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(real, condition)
            self.log("TRAIN_PatchGAN Loss", loss)
        elif optimizer_idx == 1:
            loss = self._gen_step(real, condition)
            self.log("TRAIN_Generator Loss", loss)
        return loss

    # def validation_step(self, batch, batch_idx, optimizer_idx):
    #     real, condition = batch
    #     loss = None
    #     if optimizer_idx == 0:
    #         loss = self._disc_step(real, condition)
    #         self.log("VAL_PatchGAN Loss", loss)
    #     elif optimizer_idx == 1:
    #         loss = self._gen_step(real, condition)
    #         self.log("VAL_Generator Loss", loss)
    #     return loss

    def configure_optimizers(self, lr=.0002):
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr)
        return disc_opt, gen_opt


class EpochInference(pl.callbacks.base.Callback):
    """
    Callback on each end of training epoch
    The callback will do inference on test dataloader based on corresponding checkpoints
    The results will be saved as an image with 4-rows:
        1 - Input image e.g. grayscale edged input
        2 - Ground-truth
        3 - Single inference
        4 - Mean of hundred accumulated inference
    Note that the inference have a noise factor that will generate different output on each execution
    """

    def __init__(self, dataloader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader = dataloader

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        data = next(iter(self.dataloader))
        image, target = data
        # image = image.cuda()
        # target = target.cuda()
        with torch.no_grad():
            # Take average of multiple inference as there is a random noise
            # Single
            reconstruction_init = pl_module.forward(image)
            reconstruction_init = torch.clip(reconstruction_init, 0, 1)
            # Mean
            reconstruction_mean = torch.stack([pl_module.forward(image) for _ in range(100)])
            reconstruction_mean = torch.clip(reconstruction_mean, 0, 1)
            reconstruction_mean = torch.mean(reconstruction_mean, dim=0)
        # Grayscale 1-D to 3-D
        image = torch.stack([image for _ in range(3)], dim=1)
        image = torch.squeeze(image)
        grid_image = torchvision.utils.make_grid(
            torch.cat([image, target, reconstruction_init, reconstruction_mean], dim=0), nrow=20)
        torchvision.utils.save_image(grid_image, fp=f'{trainer.log_dir}/epoch-{trainer.current_epoch:04}.png')