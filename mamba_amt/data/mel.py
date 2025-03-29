import torch
import torchaudio
from .constants import *

class MelSpectrogram(torch.nn.Module):
    def __init__(self, n_mels, sample_rate, filter_length, hop_length,
                 win_length=None, mel_fmin=0.0, mel_fmax=None):
        """
        Computes a log mel spectrogram.
        
        Parameters:
            n_mels (int): number of mel filter banks.
            sample_rate (int): audio sampling rate.
            filter_length (int): FFT size.
            hop_length (int): hop length between frames.
            win_length (int, optional): window length. If None, defaults to filter_length.
            mel_fmin (float): minimum frequency for mel filters.
            mel_fmax (float or None): maximum frequency for mel filters.
        """
        super(MelSpectrogram, self).__init__()
        if win_length is None:
            win_length = filter_length

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=filter_length,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode='constant',
            n_mels=n_mels,
            f_min=mel_fmin,
            f_max=mel_fmax,
            window_fn=torch.hann_window,   # use Hann window
            power=1.0,                     # use magnitude (not power) spectrogram
            center=True,
            norm='slaney'
        )

    def forward(self, y):
        """
        Computes the log mel spectrogram.
        
        Parameters:
            y (Tensor): audio waveform tensor of shape (B, T) with values in [-1, 1].
            
        Returns:
            Tensor: log mel spectrogram of shape (B, n_mels, time_frames)
        """
        mel_spec = self.mel_transform(y)
        # Apply log scaling (with clamping to avoid log(0))
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return log_mel_spec
