import torch
import torch.nn.functional as F
from torch import nn
from mamba_ssm import Mamba

class ConvStack(nn.Module):
    def __init__(self, input_features, output_features, dropout=0.25):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),

            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),

            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),

            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(dropout)
        )

    def forward(self, mel):
        x = self.cnn(mel.unsqueeze(1))
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class Transcriber(nn.Module):
    def __init__(self, input_features, out_features, mamba_blocks, d_model, d_state, d_conv, expand):
        super().__init__()
        self.conv_stack = ConvStack(input_features=input_features, output_features=d_model)
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(mamba_blocks)
            ])
        self.onset = nn.Linear(d_model, out_features)
        self.offset = nn.Linear(d_model, out_features)
        self.frame = nn.Linear(d_model, out_features)
        self.velocity = nn.Linear(d_model, out_features)

    def forward(self, x):
        x = self.conv_stack(x)
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)
        onset = torch.sigmoid(self.onset(x))
        offset = torch.sigmoid(self.offset(x))
        frame = torch.sigmoid(self.frame(x))
        velocity = self.velocity(x)
        return onset, offset, frame, velocity