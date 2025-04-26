import argparse
import os
import json
import torch
import librosa

from mamba_amt.models import Mamba_AMT
from mamba_amt.inference import predictions_to_midi, save_pianoroll, windowed_inference
from mamba_amt.data.constants import SAMPLE_RATE


def load_audio(audio_path, sample_rate):
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if sr != sample_rate:
        raise ValueError(f"Expected sample rate {sample_rate}, got {sr}")
    return torch.from_numpy(audio)


def load_model(ckpt_path, config_path="mamba_amt/configs/mamba_amt.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    model_config = config['model']
    model = Mamba_AMT(model_config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model.eval()
    model.to("cuda")
    return model, config['training']


def run_inference(model, audio_tensor, training_config):
    batch = {"audio": audio_tensor.unsqueeze(0)}
    sequence_length = training_config["sequence_length"] * 3
    return windowed_inference(model, batch, sequence_length)


def save_outputs(output_dir, predictions, onset_thresh, frame_thresh):
    os.makedirs(output_dir, exist_ok=True)

    midi_path = os.path.join(output_dir, "transcription.midi")
    image_path = os.path.join(output_dir, "pianoroll.png")

    predictions_to_midi(midi_path,
                        predictions['onset'],
                        predictions['frame'],
                        predictions['velocity'],
                        onset_threshold=onset_thresh,
                        frame_threshold=frame_thresh)

    save_pianoroll(image_path,
                   predictions['onset'],
                   predictions['frame'],
                   onset_threshold=onset_thresh,
                   frame_threshold=frame_thresh)

def main():
    parser = argparse.ArgumentParser(description="Transcribe piano audio using Mamba-AMT")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint .ckpt file")
    parser.add_argument("-a", "--audio", type=str, required=True, help="Path to input audio file (wav/mp3)")
    parser.add_argument("-o", "--output", type=str, default="./output", help="Directory to save outputs (MIDI and pianoroll)")
    parser.add_argument("--onset_threshold", type=float, default=0.3, help="Onset probability threshold")
    parser.add_argument("--frame_threshold", type=float, default=0.5, help="Frame probability threshold")

    args = parser.parse_args()

    model, training_config = load_model(args.ckpt)
    audio_tensor = load_audio(args.audio, SAMPLE_RATE)
    predictions = run_inference(model, audio_tensor, training_config)
    save_outputs(args.output, predictions, args.onset_threshold, args.frame_threshold)

    print(f"Done. Outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
