import os
import multiprocessing
import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import resnet


DATA_ROOT = 'data'
TRAIN_DATA = os.path.join(DATA_ROOT, 'train')
VAL_DATA = os.path.join(DATA_ROOT, 'val')
TEST_DATA = os.path.join(DATA_ROOT, 'test')

BATCH_SIZE = 64
EPOCH = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
NUM_CLASSES = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'log/dropout+layer3+layer4/best_model.pth'
LOG_DIR = 'log/test'


def load_data():
    """
    load data and transform
    Returns:
        train_loader: DataLoader
        val_loader: DataLoader
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.33)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageFolder(TRAIN_DATA, transform=train_transform)
    val_dataset = ImageFolder(VAL_DATA, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader


def load_model():
    """
    Load model from resnet.py
    """
    model = resnet.ResNet(num_classes=NUM_CLASSES, pretrained=True)
    # model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.to(DEVICE)

    return model


def train_epoch(model, train_loader, criterion, optimizer):
    """
    train the model per epoch
    """
    model.train()
    losses = []
    correct = 0
    total = 0

    for _, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = np.mean(losses)
    train_acc = correct / total

    return train_loss, train_acc


def eval_epoch(model, val_loader, criterion):
    """
    evaluate the model per epoch
    """
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = np.mean(val_losses)
    val_acc = val_correct / val_total

    return val_loss, val_acc


def print_log(epoch, log_stats, writer):
    """
    print the log 
    write the log to tensorboard
    """
    train_loss = log_stats['train_loss']
    train_acc = log_stats['train_acc']
    val_loss = log_stats['val_loss']
    val_acc = log_stats['val_acc']

    print(f'Epoch [{epoch+1}/{EPOCH}], '
          f'Train Loss: {train_loss:.4f}, '
          f'Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}')

    writer.add_scalar('Epoch/Train Loss', train_loss, epoch+1)
    writer.add_scalar('Epoch/Train Acc', train_acc, epoch+1)
    writer.add_scalar('Epoch/Val Loss', val_loss, epoch+1)
    writer.add_scalar('Epoch/Val Acc', val_acc, epoch+1)


def train(model, train_loader, val_loader):
    """
    train the model
    evaluate the model
    save the model
    """
    writer = SummaryWriter(log_dir=LOG_DIR)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters()
                    if 'fc' not in n], 'lr': LEARNING_RATE*0.3},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.7,
        patience=3,
        min_lr=5e-8
    )

    best_val_acc = 0.0
    os.makedirs(LOG_DIR, exist_ok=True)

    for epoch in range(EPOCH):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        log_stats = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        scheduler.step(val_acc)

        print_log(epoch, log_stats, writer)

        model_path = os.path.join(LOG_DIR, f"{epoch+1}_{val_acc:.3f}.pth")
        torch.save(model.state_dict(), model_path)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(LOG_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)

    writer.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_data, val_data = load_data()
    resnet50 = load_model()
    train(resnet50, train_data, val_data)
