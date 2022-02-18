import os

import numpy as np
import sklearn

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
# from torchmetrics import Accuracy


AVAIL_GPUS = min(1, torch.cuda.device_count())
batch_size = 2048 if AVAIL_GPUS else 64

def read(fname):
    array = np.load(fname)
    return array['arr_0'], array['arr_1']


class Filtering(LightningModule):
    def __init__(self) -> None:
        super().__init__()

        in_features = 24
        hidden_size = 512
        nb_layers = 3
        self.lr = 0.001
        # Construct the MLP architecture
        self.input_layer = nn.Linear(in_features, hidden_size)
        layers = [
            nn.Linear(hidden_size, hidden_size) for _ in range(nb_layers - 1)
        ]

        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.act = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        for l in self.layers:
            x = l(x)
            x = self.act(x)
            x = self.layernorm(x)  # Option of LayerNorm
        x = self.sigmoid(self.output_layer(x))
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        preds = self(x).squeeze()
        loss = F.binary_cross_entropy(preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = F.binary_cross_entropy(preds, y)
        # self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        # fpr, tpr, _ = sklearn.metrics.roc_curve(y, preds)
        # auc = sklearn.metrics.auc(fpr, tpr)

        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_auc", auc, prog_bar=True)
        # self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        fname = "/global/homes/x/xju/work/exatrkx/data/train/{}"
        train_in, train_label = read(fname.format("1003.npz"))
        train_dataset = TensorDataset(
            torch.Tensor(train_in), torch.Tensor(train_label))
        return DataLoader(train_dataset, batch_size=batch_size)

    def val_dataloader(self):
        fname = "/global/homes/x/xju/work/exatrkx/data/train/{}"
        test_in, test_label = read(fname.format("1019.npz"))
        test_dataset = TensorDataset(
            torch.Tensor(test_in), torch.Tensor(test_label))
        return DataLoader(test_dataset, batch_size=batch_size)


def train():
    model = Filtering()
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=3,
        progress_bar_refresh_rate=20,
        )

    trainer.fit(model)

train()