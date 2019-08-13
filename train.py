import numpy as np
import pandas as pd
import time
import os
import copy
import sys
import cv2

# import sys
# package_dir = "../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/"
# sys.path.insert(0, package_dir)

import pretrainedmodels

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from contrast_enhance import *

from tqdm.auto import tqdm
from PIL import Image, ImageFile

device = torch.device("cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_PATH = "./"


class RetinoDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(DATA_PATH + 'train_images',
                                self.data.loc[idx, 'id_code'] + '.png')
        image = cv2.imread(img_name)
        image = clahe_channel(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'], dtype=torch.float)
        return image, label


total_trainset = RetinoDataset(DATA_PATH + "train.csv", transform=data_transforms["train"])

train_dataset, valid_dataset = torch.utils.data.random_split(total_trainset, [3262, 400])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

dataset_sizes = [len(train_dataset), len(valid_dataset)]


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    print('-' * 10)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = -1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train()

        running_loss = 0.0
        valid_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader)):
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(device)
            labels = labels.view(-1, 1)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / dataset_sizes[0]

        model.eval()

        for i, batch in enumerate(tqdm(valid_dataloader)):
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(device)
            labels = labels.view(-1, 1)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            valid_loss += loss.item() * inputs.size(0)
        epoch_valid_loss = valid_loss / dataset_sizes[1]

        print('Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch_loss, epoch_valid_loss))

        if best_loss == -1 or epoch_valid_loss < best_loss:
            best_loss = epoch_valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model


def run():
    model_ft = pretrainedmodels.__dict__['resnet101'](num_classes=1000, pretrained='imagenet')

    for param in model_ft.parameters():
        param.requires_grad = False

    model_ft.avg_pool = nn.AdaptiveAvgPool2d(1)
    model_ft.last_linear = nn.Sequential(
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=2048, out_features=2048, bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1, bias=True),
    )
    model_ft = model_ft.to(device)

    criterion = nn.MSELoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)

    checkpoint = {
        'state_dict': model_ft.state_dict(),
        'optimizer': optimizer_ft.state_dict()}
    torch.save(checkpoint, './aptos_resnet_checkpoint_contrast_0726.pth')

    return True


if __name__ == "__main__":
    run()
