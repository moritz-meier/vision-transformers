import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vision_utils
import torch.utils.data as data
import torch.nn.functional as functional
import torch.optim as optim

import pytorch_lightning as ptl
import torchmetrics as metrics
import pytorch_lightning.callbacks as callbacks

import einops

import models


class MNIST_ViT(ptl.LightningModule):

    def __init__(
            self,
            train_dataset: datasets.MNIST,
            val_dataset: datasets.MNIST,
            *,
            patch_size=7,
            n_layers=6,
            n_heads=8,
            token_size=None,
            mlp_expansion=4,
            p_dropout=0.1,
            batch_size=100,
            lr=0.05,
            lr_momentum=0.01,
            lr_step_size=5,
            lr_gamma=0.5):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.save_hyperparameters(ignore=["train_dataset", "val_dataset"])

        self.train_acc = metrics.Accuracy()
        self.val_acc = metrics.Accuracy()

        self.model = models.VisionTransformer(img_size=28, n_classes=10, **self.hparams)

    def forward(self, x):
        self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=self.hparams.lr_momentum)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.model(x)
        loss = functional.nll_loss(y_pred, y)

        self.train_acc(y_pred, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.model(x)
        loss = functional.nll_loss(y_pred, y)

        self.val_acc(y_pred, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:

        pos_embs = self.model[0].positions[1:, :]

        def cos_sim(x):
            norm = x.norm(dim=-1).view(1, x.shape[0])
            norm_t = norm.view(x.shape[0], 1)
            scale = norm_t @ norm

            cos_sim = (x @ x.T) / scale

            return einops.rearrange(cos_sim, "h (w1 w2) -> h w1 w2", w1=int(x.shape[0]**0.5))

        pos_emb_sim = cos_sim(pos_embs)
        img = vision_utils.make_grid(einops.rearrange(pos_emb_sim, "n h w -> n () h w"), nrow=pos_emb_sim.shape[-1])

        self.logger.experiment.add_image(f"pos_emb_sim_{self.current_epoch}", img)

        return super().on_train_epoch_end()
