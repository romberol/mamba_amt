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

class BiMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, layer_idx):
        super().__init__()
        self.d_model = d_model
        
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=layer_idx)

        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):      
        forward_inp = x
        backward_inp = torch.flip(x, dims=[1])

        forward_out = self.mamba(forward_inp)
        backward_out = self.mamba(backward_inp)
        backward_out = torch.flip(backward_out, dims=[1])

        mamba_out = forward_out + backward_out
        
        mamba_out = self.norm(mamba_out)
        out = self.ff(mamba_out)

        return out


class Transcriber(nn.Module):
    def __init__(
            self, 
            input_features: int, 
            out_features: int, 
            mamba_blocks: int,
            bidirectional: bool,
            mamba_config: dict,
            use_skip: bool
    ):
        super().__init__()
        assert mamba_config is not None, "Mamba config must be provided"
        d_model = mamba_config['d_model']
        self.use_skip = use_skip

        self.conv_stack = ConvStack(input_features=input_features, output_features=d_model)
        self.conv_norm = nn.LayerNorm(d_model)

        self.mamba_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(mamba_blocks):
            if bidirectional:
                block = BiMamba(layer_idx=i, **mamba_config)
            else:
                block = Mamba(layer_idx=i, **mamba_config)
            
            self.mamba_layers.append(block)
            self.norm_layers.append(nn.LayerNorm(d_model))
        
        self.onset = nn.Linear(d_model, out_features)
        self.offset = nn.Linear(d_model, out_features)
        self.frame = nn.Linear(d_model, out_features)
        self.velocity = nn.Linear(d_model, out_features)

    def forward(self, x, **mamba_kwargs):
        x = self.conv_stack(x)
        x = self.conv_norm(x)

        for mamba_block, norm in zip(self.mamba_layers, self.norm_layers):
            residual = x if self.use_skip else None
            x = mamba_block(x, **mamba_kwargs)
            x = norm(x)
            if self.use_skip:
                x = x + residual

        onset = torch.sigmoid(self.onset(x))
        offset = torch.sigmoid(self.offset(x))
        frame = torch.sigmoid(self.frame(x))
        velocity = self.velocity(x)
        return onset, offset, frame, velocity
