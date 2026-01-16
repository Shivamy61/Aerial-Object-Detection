#IMPORT LIBRARIES
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2

 
# IMAGE TRANSFORMS
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# LOAD DATASETS
train_data = datasets.ImageFolder("train", transform=train_tfms)
valid_data = datasets.ImageFolder("valid", transform=test_tfms)
test_data  = datasets.ImageFolder("test",  transform=test_tfms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

# -----------------------------
# 3. LOAD PRETRAINED MOBILENETV2
# -----------------------------
model = mobilenet_v2(weights='IMAGENET1K_V1')

# Freeze feature layers
for param in model.features.parameters():
    param.requires_grad = False

# Change the classifier to 2 classes
model.classifier[1] = nn.Linear(model.last_channel, 2)

model = model.to(device)

# -----------------------------
# 4. LOSS FUNCTION & OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# -----------------------------
# 5. TRAIN LOOP
# -----------------------------
def train(model, loader):
    model.train()
    running_loss = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    return running_loss / len(loader), correct / len(loader.dataset)

# -----------------------------
# 6. VALIDATION LOOP
# -----------------------------
def validate(model, loader):
    model.eval()
    running_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    return running_loss / len(loader), correct / len(loader.dataset)

# -----------------------------
# 7. TRAINING PROCESS
# -----------------------------
epochs = 10

for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = validate(model, valid_loader)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}")

# -----------------------------
# 8. TESTING
# -----------------------------
test_loss, test_acc = validate(model, test_loader)
print(f"\nTest Accuracy: {test_acc:.4f}")

# -----------------------------
# 9. SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "bird_drone_mobilenetv2.pth")
print("Model saved!")

