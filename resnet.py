import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet(nn.Module):
    """
    Using resnet50 from torchvision.models
    Using pretrained weights from ResNet50_Weights 
    """
    def __init__(self, num_classes=100, pretrained=True):
        """
        Resnet initialization
        Args:
            num_classes: int
            pretrained: bool
        """
        super().__init__()

        if pretrained:
            self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.base_model = resnet50(weights=None)

        self.conv1 = self.base_model.conv1
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.avgpool = self.base_model.avgpool

        self.layer3.add_module('dropout', nn.Dropout2d(0.3))
        self.layer4.add_module('dropout', nn.Dropout2d(0.3))

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def count_params(self):
        """
        Returns:
            total params: int
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params

    def forward(self, x):
        """
        Forward pass
        Args:
            x: torch.Tensor
        Returns:
            x: torch.Tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
