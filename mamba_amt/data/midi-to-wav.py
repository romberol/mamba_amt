import pretty_midi
import numpy as np
import soundfile as sf
import os
import tqdm

def midi_to_wav(midi_path, wav_path, sample_rate=44100):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Synthesize to audio
    audio_data = midi_data.fluidsynth()
    
    # Normalize and save as WAV
    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    sf.write(wav_path, audio_data, sample_rate)

dataset_path = "datasets/maestro-v3.0.0"
years = ["2006", "2008"]

for year in years:
    print(f"Processing year {year}")
    folder = os.path.join(dataset_path, year)
    for file in tqdm.tqdm(os.listdir(folder)):
        if file.endswith(".midi"):
            midi_path = os.path.join(folder, file)
            wav_path = os.path.join(folder, file.replace(".midi", ".wav"))
            midi_to_wav(midi_path, wav_path)
