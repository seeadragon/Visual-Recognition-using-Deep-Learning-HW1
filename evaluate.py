import os
import glob
import csv
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import resnet


DATA_ROOT = 'data'
TEST_DATA = os.path.join(DATA_ROOT, 'test')
NUM_CLASSES = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'log/fc_1024/best_model.pth'


class TestDataset(Dataset):
    """
    Test dataset class for loading and processing test images
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))

        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)


        return image, img_path


def load_data():
    """
    Load test data and apply different Test-Time Augmentation (TTA) transformations
    """
    tta_transforms = [
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    ]
    dataloaders = []
    for tta_transform in tta_transforms:
        test_dataset = TestDataset(TEST_DATA, transform=tta_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        dataloaders.append(test_loader)

    return dataloaders


def load_model():
    """
    Load model from resnet.py
    """
    model = resnet.ResNet(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.to(DEVICE)
    model.eval()

    return model


def evaluate(model, test_loaders):
    """
    Evaluate the model on test data
    """
    img_probs = {}
    results = []
    with torch.no_grad():
        for loader in test_loaders:
            for _, (inputs, img_paths) in enumerate(loader):
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)

                probs = F.softmax(outputs, dim=1)

                img_name = os.path.splitext(os.path.basename(img_paths[0]))[0]

                if img_name not in img_probs:
                    img_probs[img_name] = np.zeros(NUM_CLASSES)
                img_probs[img_name] += probs.cpu().numpy().squeeze()

    for img_name, probs in img_probs.items():
        avg_probs = probs / len(test_loaders)
        pred_class = np.argmax(avg_probs)
        results.append((img_name, pred_class))

    return results


def get_class_mapping():
    """
    Mapping model index to class label
    """
    folder_names = [str(i) for i in range(NUM_CLASSES)]

    sorted_names = sorted(folder_names)

    mapping = {}
    for i, name in enumerate(sorted_names):
        mapping[i] = int(name)

    return mapping


def save_to_csv(results, output_path):
    """
    Saving the prediction result as csv
    """
    class_mapping = get_class_mapping()
    mapped_results = []
    for img_name, pred_class in results:
        mapped_class = class_mapping[pred_class]
        mapped_results.append((img_name, mapped_class))

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name','pred_label'])
        writer.writerows(mapped_results)


if __name__ == '__main__':
    test_loaders = load_data()
    model = load_model()

    results = evaluate(model, test_loaders)
    save_to_csv(results, 'prediction.csv')
    print(f'model parameters: {model.count_params()}')
