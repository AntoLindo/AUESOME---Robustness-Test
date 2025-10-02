import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from plot_utils import plot_training_curves, plot_validation_curves

import auesome_transform



def dataset_loader(IMG_SIZE,batch_size):

    #auesome_transform.transform()

    # ----------------------------
    # 2. Dataset & Loader
    # ----------------------------
    transform = auesome_transform.DFTDCTTransform(img_size=IMG_SIZE)

    train_dataset = datasets.ImageFolder("D:/MACHINE LEARNING/AIvsREAL_subset/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = datasets.ImageFolder("D:/MACHINE LEARNING/AIvsREAL_subset/test", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return train_loader, val_loader
