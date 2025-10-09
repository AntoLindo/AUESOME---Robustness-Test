import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from auesome_validation import validation
from auesome_train import train
#from auesome_transform import transform
from auesome_dataset_loader import dataset_loader
from auesome_network_loader import network_loader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
from show_predictions import show_predictions

import tkinter as tk
from tkinter import filedialog, Label, Button
import os

import auesome_transform


IMG_SIZE = 512

param_load_dir = "D:/MACHINE LEARNING/savings/AUESOME_rob_test_PARAMETERS_30sept.pth"






device, model = network_loader(param_load_dir)

model.eval()  # important!

transform = auesome_transform.DFTDCTTransform(img_size=IMG_SIZE)

class_names = ["fake", "real"]  # <-- attenzione all'ordine delle cartelle

# ----------------------------
# 3. Funzioni GUI
# ----------------------------
def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not file_path:
        return

    # Mostra immagine originale
    img = Image.open(file_path).convert("RGB")
    img_resized = img.resize((512, 512))
    tk_img = ImageTk.PhotoImage(img_resized)
    panel.config(image=tk_img)
    panel.image = tk_img

    # Prepara immagine per la rete
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        pred_class = class_names[pred.item()]

    # Mostra risultato
    result_label.config(text=f"Prediction: {pred_class.upper()}")

# ----------------------------
# 4. Costruisci GUI
# ----------------------------
root = tk.Tk()
root.title("Real vs Fake Classifier")

btn = Button(root, text="Select Image", command=open_image)
btn.pack()

panel = Label(root)
panel.pack()

result_label = Label(root, text="Prediction: ---", font=("Arial", 14))
result_label.pack()

root.mainloop()
