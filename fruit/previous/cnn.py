"""LEGO 表格資料的 DNN 回歸模型；原 CNN 與註解完整保存於 legacy/cnn.py。"""
from torch import nn

class LegoDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                     nn.Dropout(0.1), nn.Linear(128, 64),
                                     nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)
