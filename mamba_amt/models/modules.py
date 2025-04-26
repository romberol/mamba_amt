import torch
import torch.nn.functional as F
from torch import nn
from mamba_ssm import Mamba

class ConvStack(nn.Module):
    """
    Convolutional stack for feature extraction from mel spectrograms.

    Args:
        input_features (int): Number of input features (e.g., mel spectrogram features).
        output_features (int): Number of output features (e.g., model dimension).
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, input_features, output_features, dropout=0.1):
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

    
class MambaBlock(nn.Module):
    def __init__(self, mamba_config, bidirectional_cfg, with_ff):
        """
        Args:
        - mamba_config (dict): Configuration dictionary for Mamba layer initialization.
        - bidirectional_cfg (dict, optional): Configuration for bidirectional processing.
            Possible keys:
                - 'shared' (bool): Whether forward and backward Mamba layers share weights.
                - 'concat' (bool): Whether to concatenate forward and backward outputs.
            Defaults to None (unidirectional).
        - with_ff (bool, optional): Whether to include a feed-forward network after the Mamba layer.
            Defaults to True.
        """
        super().__init__()
        assert mamba_config is not None, "Mamba config must be provided"
        self.d_model = mamba_config['d_model']

        self.mamba_forward = Mamba(**mamba_config)

        self.bidirectional_cfg = bidirectional_cfg or {}
        self.bidirectional = bool(bidirectional_cfg)

        if self.bidirectional:
            if self.bidirectional_cfg.get('shared', False):
                self.mamba_backward = self.mamba_forward
            else:
                self.mamba_backward = Mamba(**mamba_config)

        self.pre_norm = nn.LayerNorm(self.d_model)

        self.with_ff = with_ff
        if self.with_ff:
            ff_input_dim = self.d_model * 2 if self.bidirectional_cfg.get('concat', False) else self.d_model
            self.ff = nn.Sequential(
                nn.LayerNorm(ff_input_dim),
                nn.Linear(ff_input_dim, self.d_model * 4),
                nn.GELU(),
                nn.Linear(self.d_model * 4, self.d_model)
            )

    def _apply_mamba(self, x):
        if self.bidirectional:
            forward_out = self.mamba_forward(x)
            backward_out = self.mamba_backward(torch.flip(x, dims=[1]))
            backward_out = torch.flip(backward_out, dims=[1])

            if self.bidirectional_cfg.get('concat', False):
                return torch.cat([forward_out, backward_out], dim=-1)
            else:
                return forward_out + backward_out
        else:
            return self.mamba_forward(x)

    def forward(self, x):
        residual = x
        x_norm = self.pre_norm(x)

        x_out = self._apply_mamba(x_norm)

        if self.bidirectional_cfg.get('concat', False):
            residual = residual.repeat(1, 1, 2)

        if self.with_ff:
            x_out = self.ff(x_out + residual)

        return x_out


class Transcriber(nn.Module):
    """
    Transcriber model for piano transcription using Mamba blocks.
    
    Args:
        input_features (int): Number of input features (e.g., mel spectrogram features).
        out_features (int): Number of output features (e.g., number of MIDI notes).
        mamba_blocks (int): Number of Mamba blocks to stack.
        bidirectional_cfg (dict): Configuration for bidirectional processing.
            Possible keys:
                - 'shared' (bool): Whether forward and backward Mamba layers share weights.
                - 'concat' (bool): Whether to concatenate forward and backward outputs.
        mamba_config (dict): Configuration dictionary for Mamba layer initialization.
        use_skip (bool): Whether to use skip connections in the Mamba blocks.
        with_ff (bool): Whether to include a feed-forward network after the Mamba layer.
    """
    def __init__(
            self, 
            input_features: int, 
            out_features: int, 
            mamba_blocks: int,
            bidirectional_cfg: dict,
            mamba_config: dict,
            use_skip: bool,
            with_ff: bool,
    ):
        super().__init__()
        assert mamba_config is not None, "Mamba config must be provided"
        d_model = mamba_config['d_model']
        self.use_skip = use_skip

        self.conv_stack = ConvStack(input_features=input_features, output_features=d_model)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(mamba_config=mamba_config, bidirectional_cfg=bidirectional_cfg, with_ff=with_ff)
            for _ in range(mamba_blocks)
        ])

        self.onset = nn.Sequential(
            nn.LayerNorm(d_model),
            Mamba(**mamba_config),
            nn.Linear(d_model, out_features)
        )
        self.offset = nn.Sequential(
            nn.LayerNorm(d_model),
            Mamba(**mamba_config),
            nn.Linear(d_model, out_features)
        )
        self.frame = nn.Sequential(
            nn.LayerNorm(d_model),
            Mamba(**mamba_config),
            nn.Linear(d_model, out_features)
        )
        self.velocity = nn.Sequential(
            nn.LayerNorm(d_model),
            Mamba(**mamba_config),
            nn.Linear(d_model, out_features)
        )

    def forward(self, x, **mamba_kwargs):
        x = self.conv_stack(x)

        for mamba_block in self.mamba_layers:
            residual = x if self.use_skip else None
            x = mamba_block(x, **mamba_kwargs)
            if self.use_skip:
                x = x + residual

        onset = torch.sigmoid(self.onset(x))
        offset = torch.sigmoid(self.offset(x))
        frame = torch.sigmoid(self.frame(x))
        velocity = self.velocity(x)
        return onset, offset, frame, velocity
