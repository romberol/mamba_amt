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
    
    def run_on_batch(self, batch, log_name='loss'):
        audio = batch['audio'][:, :-1]
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']
        
        onset_pred, offset_pred, frame_pred, velocity_pred = self(audio)

        predicitons = {
            'onset': onset_pred,
            'offset': offset_pred,
            'frame': frame_pred,
            'velocity': velocity_pred
        }
        
        losses = {
            f'{log_name}/onset': F.binary_cross_entropy(onset_pred, onset_label),
            f'{log_name}/offset': F.binary_cross_entropy(offset_pred, offset_label),
            f'{log_name}/frame': F.binary_cross_entropy(frame_pred, frame_label),
            f'{log_name}/velocity': F.mse_loss(velocity_pred, velocity_label)
        }

        return predicitons, losses

    def training_step(self, batch, batch_idx):
        _, losses = self.run_on_batch(batch, log_name='train_loss')
        total_loss = sum(losses.values())
        
        self.log_dict(losses, prog_bar=True, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        _, losses = self.run_on_batch(batch, log_name='val_loss')        
        self.log_dict(losses, prog_bar=True, on_step=False, on_epoch=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)