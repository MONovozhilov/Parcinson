import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, acoustic_feature_size=123, dropout_rate=0.3): # 🆕 Добавлен dropout_rate
        super(HybridModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.rnn = nn.LSTM(256 * 4, 128, 2, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.mkl = nn.Sequential(nn.Linear(128 * 2 + 256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(dropout_rate))
        self.mlp = nn.Sequential(
            nn.Linear(512 + acoustic_feature_size, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Linear(128, 2)
    
    def forward(self, spec, aco):
        cnn_out = self.cnn(spec)
        rnn_in = cnn_out.permute(0, 3, 1, 2).contiguous().view(cnn_out.size(0), cnn_out.size(3), -1)
        rnn_out, _ = self.rnn(rnn_in)
        fused = torch.cat([rnn_out[:, -1, :], cnn_out.view(cnn_out.size(0), -1)], dim=1)
        return self.classifier(self.mlp(torch.cat([self.mkl(fused), aco], dim=1)))