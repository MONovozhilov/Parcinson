import torch
import torch.nn as nn


class HybridModel(nn.Module):
    def __init__(self, acoustic_feature_size=123, n_mels=128, target_length=437):
        super(HybridModel, self).__init__()
        
        # CNN для спектрограмм
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # LSTM для временных зависимостей
        self.rnn = nn.LSTM(
            input_size=256 * 4,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Мультимодальное слияние признаков (MKL)
        self.mkl = nn.Sequential(
            nn.Linear(128 * 2 + 256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # MLP с акустическими признаками
        self.mlp = nn.Sequential(
            nn.Linear(512 + acoustic_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Классификатор
        self.classifier = nn.Linear(128, 2)
    
    def forward(self, x_spectrogram, x_acoustic):
        """
        Прямой проход через модель
        
        Аргументы:
            x_spectrogram: torch.Tensor (batch, 1, n_mels, time), спектрограммы
            x_acoustic: torch.Tensor (batch, acoustic_features), акустические признаки
        
        Возвращает:
            torch.Tensor (batch, num_classes), логиты классов
        """
        # CNN
        cnn_out = self.cnn(x_spectrogram)
        
        # RNN
        rnn_input = cnn_out.permute(0, 3, 1, 2).contiguous().view(cnn_out.size(0), cnn_out.size(3), -1)
        rnn_out, _ = self.rnn(rnn_input)
        rnn_last = rnn_out[:, -1, :]
        
        # Слияние признаков
        cnn_flat = cnn_out.view(cnn_out.size(0), -1)
        fused = torch.cat([rnn_last, cnn_flat], dim=1)
        mkl_out = self.mkl(fused)
        
        # Объединение с акустическими признаками
        combined = torch.cat([mkl_out, x_acoustic], dim=1)
        mlp_out = self.mlp(combined)
        
        # Классификация
        return self.classifier(mlp_out)