# 4 layers frozen
# sgd 
# scheduler
# augmentation and preprocessing


import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
import numpy as np
from torchvision.transforms.v2 import MixUp, CutMix

def find_bad_images(root):
    bad = []
    for folder, _, files in os.walk(root):
        for file in files:
            path = os.path.join(folder, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                bad.append(path)
    return bad

def soft_cross_entropy(pred, soft_targets):
    log_probs = torch.log_softmax(pred, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()

def main():
    print("Torch version:", torch.__version__)

    data_root = r"C:\Migration files\black hole\new bylaw\spring 26\ai\project\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\train"

    bad_files = find_bad_images(data_root)
    print("Bad files:", len(bad_files))
    for f in bad_files:
        os.remove(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    num_classes = 17
    model.fc = nn.Linear(model.fc.in_features, num_classes)


    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = False

    model = model.to(device)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    full_dataset = datasets.ImageFolder(root=data_root)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

  
    mixup = MixUp(num_classes=num_classes, alpha=0.4)
    cutmix = CutMix(num_classes=num_classes, alpha=1.0)

    num_epochs = 20
    best_val_loss = 1000
    best_val_acc=0.0

    print("Starting training...\n")

    

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

       
            if np.random.rand() < 0.5:
                images, labels = mixup(images, labels)
            else:
                images, labels = cutmix(images, labels)

            outputs = model(images)

            loss = soft_cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

           
            preds = outputs.argmax(dim=1)
            true = labels.argmax(dim=1)

            total += true.size(0)
            correct += (preds == true).sum().item()

        train_acc = 100 * correct / total

        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                loss = nn.CrossEntropyLoss()(outputs, labels)

                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {running_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
     
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_resnet50.pth")
            print(f"✓ Saved best model (val_acc={val_acc:.2f}%)")

    print("\nTraining complete.")
    print("Best validation accuracy:", best_val_acc)

if __name__ == "__main__":
    main()