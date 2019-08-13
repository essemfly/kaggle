import pandas as pd
import time
import os
import copy
import pretrainedmodels

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch_tfms import data_transforms

from tqdm.auto import tqdm
from PIL import Image, ImageFile

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RetinoTestDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('./test_images',
                                self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        return image


test_dataset = RetinoTestDataset("./test.csv", transform=data_transforms["test"])
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
