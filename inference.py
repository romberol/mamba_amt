from mamba_amt.data import MAESTRO
from mamba_amt.models import AMT_Trainer
from mamba_amt.inference import predictions_to_midi, save_pianoroll
import soundfile as sf
import torch
import os
import time

train_dataset = MAESTRO(path="datasets/maestro-v3.0.0", sequence_length=327680, groups=['2008'])

sample = train_dataset[1]
audio = sample["audio"][:-1]

model_config = {
    'mamba_blocks': 2,  # number of mamba blocks
    'd_model': 256,     # model dimension
    'd_state': 64,      # B anc C dimensions
    'd_conv': 4,        # local convolution width
    'expand': 2 ,       # block expansion factor
    'out_features': 88
}

out_folder = "output"
os.makedirs(out_folder, exist_ok=True)

ckpt_path = "piano-transcription/najpvkpn/checkpoints/epoch=599-step=4800.ckpt"
model = AMT_Trainer.load_from_checkpoint(ckpt_path, model_config=model_config, lr=1e-3)

with torch.no_grad():
    start = time.time()
    onset_pred, offset_pred, frame_pred, velocity_pred = model(audio.unsqueeze(0))
    print(f"Time taken: {time.time() - start:.2f}s")

sf.write(os.path.join(out_folder, "sample.wav"), audio.cpu().numpy(), 16000)
save_pianoroll(os.path.join(out_folder, "sample_pianoroll.png"), sample["onset"], sample["frame"])

predictions_to_midi(os.path.join(out_folder, "predictions.midi"), 
                    onset_pred, 
                    frame_pred, 
                    velocity_pred, 
                    onset_threshold=0.5, 
                    frame_threshold=0.5)
save_pianoroll(os.path.join(out_folder, "predictions.png"), onset_pred[0], frame_pred[0], onset_threshold=0.5, frame_threshold=0.5)
