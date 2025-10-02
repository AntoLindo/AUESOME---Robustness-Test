import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from plot_utils import plot_training_curves, plot_validation_curves



def network_loader(param_load_dir):

    # ----------------------------
    # 3. ResNet18 with 9 input channels
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)

    # Modify first conv to accept 9 channels instead of 3
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=9,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # Copy pretrained weights (repeat them to 9 channels)
    with torch.no_grad():
        model.conv1.weight = nn.Parameter(
            torch.cat([old_conv.weight] * 3, dim=1)[:, :9, :, :]
        )

    # Replace last FC
    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    #model = model.to(device)

    # Carica lo stato salvato
    model.load_state_dict(torch.load(param_load_dir))
    model = model.to(device)

    return device, model
