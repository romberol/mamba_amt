import pytorch_lightning as pl
import torch
from ..data import MelSpectrogram
from ..data.constants import *
from .modules import Transcriber
import torch.nn.functional as F

class Mamba_AMT(pl.LightningModule):
    def __init__(self, model_config, start_lr=5e-4, end_lr=5e-5, max_epochs=100):
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
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.max_epochs = max_epochs

    def forward(self, audio, **mamba_kwargs):
        mel = self.melspectrogram(audio).transpose(-1, -2)
        return self.model(mel, **mamba_kwargs)
    
    def run_on_batch(self, batch, log_name='loss'):
        audio = batch['audio'][:, :-1].to(self.device)
        onset_label = batch['onset'].to(self.device)
        offset_label = batch['offset'].to(self.device)
        frame_label = batch['frame'].to(self.device)
        velocity_label = batch['velocity'].to(self.device)
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=self.end_lr / self.start_lr, total_iters=self.max_epochs
        )
        return [optimizer], [scheduler]
