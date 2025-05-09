import os
from abc import abstractmethod
from glob import glob

import numpy as np
import mido
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import *
import librosa
from torch.utils.data import DataLoader

from audiomentations import Compose, SevenBandParametricEQ, PitchShift, AddBackgroundNoise, PolarityInversion, ApplyImpulseResponse
import pandas as pd


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, augment=False,
                 noise_path=None, ir_path=None):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.augment = augment

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))
        
        if self.augment:
            self.augmentations = Compose([
                SevenBandParametricEQ(min_gain_db=-10, max_gain_db=5, p=0.5),
                PitchShift(min_semitones=-0.1, max_semitones=0.1, p=0.5),
                AddBackgroundNoise(
                    sounds_path=noise_path,
                    min_snr_db=17.5,
                    max_snr_db=25,
                    noise_transform=PolarityInversion(),
                    p=0.5
                ),
                SevenBandParametricEQ(min_gain_db=-10, max_gain_db=5, p=0.5),
                ApplyImpulseResponse(ir_path=ir_path, p=0.5)
            ])

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end]
            result['label'] = data['label'][step_begin:step_end, :]
            result['velocity'] = data['velocity'][step_begin:step_end, :]
        else:
            result['audio'] = data['audio']
            result['label'] = data['label']
            result['velocity'] = data['velocity']

        if self.augment:
            augmented_audio = self.augmentations(samples=result['audio'].numpy().astype(np.float32) / 32768.0, sample_rate=SAMPLE_RATE)
            result['audio'] = torch.from_numpy(augmented_audio)
        else:
            result['audio'] = result['audio'].float().div_(32768.0)
        
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        audio, sr = soundfile.read(audio_path, dtype='int16')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.int16)  # Convert stereo to mono
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.int16)

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3  # onset region (beginning of the note)
            label[onset_right:frame_right, f] = 2  # sustained part of the note (while the note is held) 
            label[frame_right:offset_right, f] = 1  # offset region (end of the note)
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='data/MAESTRO', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, 
                 augment=False, noise_path=None, ir_path=None):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, device, 
                         augment, noise_path, ir_path)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        print(f"Loading {group} group {self.available_groups()}")
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = pd.read_csv(os.path.join(self.path, 'maestro-v3.0.0.csv'))
            files = sorted([
                (os.path.join(self.path, row.audio_filename), os.path.join(self.path, row.midi_filename))
                for _, row in metadata[metadata.split == group].iterrows()
            ])

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


# https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/midi.py
def parse_midi(path):
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = mido.MidiFile(path)

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
            events.append(event)

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(n for n in events[offset['index'] + 1:]
                          if n['type'] == 'sustain_off' or n['note'] == onset['note'] or n is events[-1])

        note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    return np.array(notes)


if __name__ == "__main__":
    dataset = MAESTRO(path="datasets/maestro-v3.0.0", sequence_length=327680, groups=['2006'], augment=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    print(len(loader))
    sample = next(iter(loader))
