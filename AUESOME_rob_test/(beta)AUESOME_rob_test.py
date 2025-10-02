'''
TO ACTIVATE CUDA ENVIRMENT: conda activate pytorch_cuda12
'''

import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from plot_utils import plot_training_curves, plot_validation_curves
from auesome_validation import validation
from auesome_train import train
#from auesome_transform import transform
from auesome_dataset_loader import dataset_loader
from auesome_network_loader import network_loader




IMG_SIZE = 200
EPOCHS = 1
batch_size = 8
param_load_dir = "D:/MACHINE LEARNING/savings/AUESOME_rob_test_PARAMETERS_30sept.pth"
param_save_dir = "D:/MACHINE LEARNING/savings/AUESOME_rob_test_PARAMETERS_1oct.pth"






train_loader, val_loader = dataset_loader(IMG_SIZE,batch_size)

device, model = network_loader(param_load_dir)


train(train_loader,EPOCHS,device,model)

validation(val_loader,EPOCHS,device,model)

# ----------------------------
# 6. Save model
# ----------------------------
torch.save(model.state_dict(), param_save_dir)
print("✅ Model saved as ")
print(param_save_dir)
