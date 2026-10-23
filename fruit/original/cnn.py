# cnn.py
# 簡潔穩定的 CNN，適用 Fashion-MNIST (1×28×28)
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
     def __init__(self, num_classes: int = 10):
        super().__init__()
        # 28x28 → 14x14 → 7x7
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28→14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14→7
            
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # # 64×7×7 = 3136
            nn.Linear(64*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        
     def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # 快速自測
    m = CNN()
    x = torch.randn(8, 1, 28, 28)
    y = m(x)
    print(y.shape)  # [8, 10]
