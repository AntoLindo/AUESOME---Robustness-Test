import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary

# --- 1. Trasformazioni immagini ---
IMG_SIZE = 512
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # mean ImageNet
                         [0.229, 0.224, 0.225])  # std ImageNet
])

# --- 2. Caricamento dataset ---
train_dataset = datasets.ImageFolder("D:/MACHINE LEARNING/AIvsREAL_subset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder("D:/MACHINE LEARNING/AIvsREAL_subset/test", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- 3. Caricamento ResNet18 pre-addestrata ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)

# --- 4. Adattare l'ultimo layer ---
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# --- 5. Loss e optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- 6. Training con monitoraggio ---
EPOCHS = 1
for epoch in range(EPOCHS):
    # Training
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

# --- 7. Salvataggio modello ---
torch.save(model.state_dict(), "D:/MACHINE LEARNING/savings/resnet18_ai_vs_real.pth")

print("✅ Model saved as resnet18_ai_vs_real.pth")

# --- 8. Summary finale ---
summary(model, (3, 512, 512))
