import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
from show_predictions import show_predictions

# ----------------------------
# 1. Custom Transform (come in training)
# ----------------------------
def normalize_channel(channel):
    ch = channel - np.min(channel)
    ch = ch / (np.max(ch) + 1e-8)
    ch = (ch * 255).astype(np.uint8)
    return ch

def make_rgb_image(channels):
    return np.stack([normalize_channel(ch) for ch in channels], axis=-1)

class DFTDCTTransform:
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, img: Image.Image):
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img)

        magnitude_spectra, phases, dct_channels = [], [], []

        for c in range(3):
            channel = np.float32(img[:, :, c])

            # DFT
            dft = cv2.dft(channel, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            mag = 50 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
            phase = cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1])

            # DCT
            dct = cv2.dct(channel)
            dct_log = np.log(np.abs(dct) + 1)

            magnitude_spectra.append(mag)
            phases.append(phase)
            dct_channels.append(dct_log)

        img_dft_mag = make_rgb_image(magnitude_spectra)
        img_dft_phase = make_rgb_image(phases)
        img_dct_mag = make_rgb_image(dct_channels)

        merged = np.concatenate([img_dft_mag, img_dft_phase, img_dct_mag], axis=-1)
        merged = merged.astype(np.float32) / 255.0
        merged = np.transpose(merged, (2, 0, 1))

        return torch.tensor(merged, dtype=torch.float)

# ----------------------------
# 2. Dataset & Loader
# ----------------------------
IMG_SIZE = 224
transform = DFTDCTTransform(img_size=IMG_SIZE)

test_dataset = datasets.ImageFolder("D:/MACHINE LEARNING/AIvsREAL_subset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

# ----------------------------
# 3. Load 9-channel ResNet18
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)

# Modify conv1 to accept 9 channels
old_conv = model.conv1
model.conv1 = nn.Conv2d(
    in_channels=9,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None
)

# Replace FC
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load saved weights
model.load_state_dict(torch.load("D:/MACHINE LEARNING/savings/resnet18_dftdct.pth", map_location=device))
model = model.to(device)
#model.eval()  # important!

# ----------------------------
# 4. Evaluation
# ----------------------------
y_true, y_pred = [], []
correct, total = 0, 0
'''
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images) #Passa il batch attraverso la rete. outputs è un tensore di dimensione (batch_size, num_classes) con i logits (punteggi grezzi prima della softmax).
        _, preds = torch.max(outputs, 1) #_ takes the index of the current image, while, preds takes the value of the real prediction, labeled 1. This finds all the (1) predictions
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ----------------------------
# 5. Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap="Blues")
plt.show()
'''
# Mostra 8 immagini con predizioni
show_predictions(test_dataset, test_loader, model, device, num_images=30, row_size=10)
