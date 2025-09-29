import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from plot_utils import plot_training_curves



# ----------------------------
# 1. Custom Transform
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
        # Convert PIL → numpy RGB
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

        # Stack: (9, H, W)
        img_dft_mag = make_rgb_image(magnitude_spectra)
        img_dft_phase = make_rgb_image(phases)
        img_dct_mag = make_rgb_image(dct_channels)

        merged = np.concatenate([img_dft_mag, img_dft_phase, img_dct_mag], axis=-1)  # (H, W, 9)
        merged = merged.astype(np.float32) / 255.0  # normalize 0-1
        merged = np.transpose(merged, (2, 0, 1))  # (C, H, W)

        return torch.tensor(merged, dtype=torch.float)

# ----------------------------
# 2. Dataset & Loader
# ----------------------------
IMG_SIZE = 224
transform = DFTDCTTransform(img_size=IMG_SIZE)

train_dataset = datasets.ImageFolder("D:/MACHINE LEARNING/AIvsREAL_subset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = datasets.ImageFolder("D:/MACHINE LEARNING/AIvsREAL_subset/test", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

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
model = model.to(device)

# ----------------------------
# 4. Loss & Optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------
# 5. Training Loop
# ----------------------------
EPOCHS = 1
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

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# ----------------------------
# 6. Save model
# ----------------------------
torch.save(model.state_dict(), "D:/MACHINE LEARNING/savings/resnet18_dftdct.pth")
print("✅ Model saved as resnet18_dftdct.pth")


# ----------------------------
# 7. Plot Loss and accuracy variatios
# ----------------------------


plot_training_curves(train_losses, val_losses, train_accs, val_accs)
