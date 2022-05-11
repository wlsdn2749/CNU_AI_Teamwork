from cProfile import label
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from PIL import Image
import itertools
    

class ImgDataset(Dataset):
    def __init__(self, df, images_folder, transform = None):
        self.df = df
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = {'bag':0, 'bed':1, 'chair':2, 'coffeetable':3, 'cup':4, 'kitchentools':5, 'lamp':6, 'laptop':7, 'LivingSofa':8, 'pot':9, 'shoe':10}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        ####################################################
        # 입력 받은 csv를 통해 데이터셋을 구현하고
        # 이비지 형태를 변환하는 코드 구현

        ####################################################
        return image, label


class CNN(pl.LightningModule):
    def __init__(self, batch_size, image_size):
        super(CNN, self).__init__()

        self.transform = transforms.Compose([
        ####################################################
        # 이미지 형태를 tensor로 변환하기 위한 부분
        # torchvision.transforms 참고

        ####################################################
        ])
        self.batch_size = batch_size

        ####################################################
        # 모델에 사용될 layer들을 구현

        ####################################################

    def forward(self, x):
        batch_size = x.size(0)

        ####################################################
        # 모델의 layer를 활용해 모델 구조를 구현

        ####################################################
        return output

    def prepare_data(self):
        train_df = pd.read_csv('train_data.csv')
        train_dataset = ImgDataset(train_df, 'train', self.transform)

        # split dataset to train, val, test ! ==> 0.8 / 0.2
        train_dataset, val_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.8), (len(train_dataset) - int(len(train_dataset) * 0.8))])
        
        valid_df = pd.read_csv('val_data.csv')
        test_dataset = ImgDataset(valid_df, 'val', self.transform)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=3e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch

        output = self(data)
        loss = F.cross_entropy(output, target)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch

        output = self(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {"val_loss": loss, "correct": correct}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log("val_loss", avg_loss, prog_bar=True)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        data, target = batch

        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {"correct": correct}

    def test_epoch_end(self, outputs):
        all_correct = sum([output["correct"] for output in outputs])
        accuracy = all_correct / len(self.test_dataloader().dataset)
        self.log("Accuracy", accuracy)
        return accuracy

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data, target = batch
        output = self(data)
        y_hat = output.argmax(dim=1, keepdim=True)
        return y_hat


if __name__ == "__main__":
    pl.seed_everything(42)  # set seed
    torch.manual_seed(42)

    cnn_model = CNN(batch_size=32, image_size=256)

    cnn_trainer = pl.Trainer(gpus=1, max_epochs=20)
    cnn_trainer.fit(cnn_model)

    cnn_trainer.test()

    ####################################################
    # 학습된 모델을 저장하는 코드 구현
    
    ####################################################

    ####################################################
    # 저장된 모델을 불러오고
    # prediction 데이터셋을 생성해 불러온 모델에 입력하여
    # prediction 결과를 출력하는 코드 구현
    
    ####################################################

    ####################################################
    # prediction 결과를 csv파일로 저장하는 코드 구현
    # Pandas

    ####################################################


