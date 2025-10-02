import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from plot_utils import plot_training_curves, plot_validation_curves



def train(train_loader,EPOCHS,device,model):

    # ----------------------------
    # 4. Loss & Optimizer
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ----------------------------
    # 5. Training Loop
    # ----------------------------

    train_losses = []
    #val_losses = []
    train_accs = []
    #val_accs = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total


        # 🔹 Salva metriche nelle liste
        train_losses.append(train_loss)

        train_accs.append(train_acc)


        # 7. Plot Loss and accuracy variatios

        plot_training_curves(train_losses, train_accs)

        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ")
