"""水果 RGB 影像分類；原課堂程式及註解完整保留於 original/。"""
from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        layers = []
        channels = 3
        for width in (16, 32, 64):
            layers += [nn.Conv2d(channels, width, 3, padding=1),
                       nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)),
                                        nn.Flatten(), nn.Dropout(.3),
                                        nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))
