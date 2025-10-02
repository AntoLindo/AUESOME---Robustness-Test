import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from plot_utils import plot_training_curves, plot_validation_curves


def validation(val_loader,EPOCHS,device,model):



    # ----------------------------
    # 4. Loss & Optimizer
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    
    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {EPOCHS} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    '''
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    plot_validation_curves(val_losses, val_accs)'''

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
