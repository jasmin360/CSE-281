# unfrozen
# sgd smoothing
# scheduler
# changed per layer lr parameter


import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
from torchvision import transforms

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


    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)

    preprocess = weights.transforms()

    dataset = datasets.ImageFolder(
        root=data_root,
        transform=preprocess
    )


    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    print("train size:", train_size)
    print("val size", val_size)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD([
    {"params": model.layer4.parameters(), "lr": 1e-1},   
    {"params": model.fc.parameters(),     "lr": 1e-2},
    {"params": model.layer1.parameters(), "lr": 1e-3},  
    {"params": model.layer2.parameters(), "lr": 1e-3},
    {"params": model.layer3.parameters(), "lr": 1e-4},
    ], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max', 
    factor=0.5,
    patience=2  #epoch wait 
          )

    num_epochs = 10
    best_val_acc = 0.0

    print("Starting training...\n")

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {running_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet50.pth")
            print(f"✓ Saved best model (val_acc={val_acc:.2f}%)")

    print("\nTraining complete.")
    print("Best validation accuracy:", best_val_acc)


if __name__ == "__main__":
    main()