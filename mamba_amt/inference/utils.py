import torch
from mamba_amt.data.constants import HOP_LENGTH
from tqdm import tqdm

def windowed_inference(model, batch, window_size, overlap_ratio=0.5, show_progress=True):
    audio = batch['audio']
    audio_len = audio.shape[-1]

    window_size = window_size // HOP_LENGTH * HOP_LENGTH
    overlap = int(window_size * overlap_ratio)
    step = window_size - overlap

    all_predictions = {'onset': [], 'offset': [], 'frame': [], 'velocity': []}

    combined_predictions = {}
    for key in all_predictions:
        combined_predictions[key] = torch.zeros((audio_len // HOP_LENGTH + 1, 88), device=model.device)
        weight_sum = torch.zeros((audio_len // HOP_LENGTH + 1, 88), device=model.device)

    iterator = tqdm(range(0, audio_len, step)) if show_progress else range(0, audio_len, step)
    for start in iterator:
        end = min(start + window_size, audio_len)
        segment = audio[..., start:end].to(model.device)

        with torch.no_grad():
            preds = model(segment)

        segment_len_frames = preds['frame'].shape[1]
        frame_start = start // HOP_LENGTH
        frame_end = frame_start + segment_len_frames

        for key in all_predictions:
            combined_predictions[key][frame_start:frame_end] += preds[key][0]
        weight_sum[frame_start:frame_end] += 1

    for key in combined_predictions:
        combined_predictions[key] /= weight_sum

    return combined_predictions