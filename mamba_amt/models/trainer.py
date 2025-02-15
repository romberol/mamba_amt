import pytorch_lightning as pl
import torch
from ..data import MelSpectrogram
from ..data.constants import *
from .modules import Transcriber
import torch.nn.functional as F

class AMT_Trainer(pl.LightningModule):
    def __init__(self, model_config, lr):
        super().__init__()
        self.melspectrogram = MelSpectrogram(
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            filter_length=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX
        )

        model_config['input_features'] = N_MELS
        self.model = Transcriber(**model_config)
        self.lr = lr

    def forward(self, audio):
        mel = self.melspectrogram(audio).transpose(-1, -2)
        return self.model(mel)

    def training_step(self, batch, batch_idx):
        audio = batch['audio'][:, :-1]
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']
        
        onset_pred, offset_pred, frame_pred, velocity_pred = self(audio)
        
        losses = {
            'loss/onset': F.binary_cross_entropy(onset_pred, onset_label),
            'loss/offset': F.binary_cross_entropy(offset_pred, offset_label),
            'loss/frame': F.binary_cross_entropy(frame_pred, frame_label),
            'loss/velocity': F.mse_loss(velocity_pred, velocity_label)
        }
        
        total_loss = sum(losses.values())
        
        self.log_dict(losses, prog_bar=True, on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)