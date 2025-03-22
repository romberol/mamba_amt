import torch
from mido import MidiFile, MidiTrack, Message
import numpy as np
from ..data.constants import *
from PIL import Image


def extract_notes(onsets, frames, velocity, onset_threshold=0.5, frame_threshold=0.5):
    """
    Extracts note timings from onset and frame predictions.

    Args:
        onsets (torch.FloatTensor): Onset activations [frames, bins].
        frames (torch.FloatTensor): Frame activations [frames, bins].
        velocity (torch.FloatTensor): Velocity values [frames, bins].
        onset_threshold (float, optional): Threshold for onset activation.
        frame_threshold (float, optional): Threshold for frame activation.

    Returns:
        tuple: (pitches, intervals, velocities)
    """
    if onsets.ndim == 3:
        onsets = onsets.squeeze(0)
        frames = frames.squeeze(0)
        velocity = velocity.squeeze(0)
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    velocities = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

    return np.array(pitches), np.array(intervals), np.array(velocities)

def notes_to_frames(pitches, intervals, shape):
    """
    Converts note timings to frame-wise representation.

    Args:
        pitches (ndarray): Note pitches.
        intervals (list): Onset and offset times.
        shape (tuple): Shape of the frame-wise representation.

    Returns:
        tuple: (time, freqs)
    """
    if len(shape) == 3:
        shape = shape[1:]
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs


def midi_to_hz(midi):
    """
    Converts MIDI note to Hz.
    
    Args:
        midi (float or ndarray): MIDI note value(s).

    Returns:
        float or ndarray: Frequency in Hz.
    """
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

def hz_to_midi(freqs):
    """
    Converts frequency (Hz) to MIDI note.
    
    Args:
        freqs (float or ndarray): Frequency value(s) in Hz.

    Returns:
        float or ndarray: MIDI note number(s).
    """
    return 12.0 * (np.log2(freqs) - np.log2(440.0)) + 69.0


def save_midi(path, pitches, intervals, velocities):
    """
    Saves extracted notes as a MIDI file.
    
    Args:
        path (str): Path to save the MIDI file.
        pitches (ndarray): Note pitches.
        intervals (list): Onset and offset times.
        velocities (list): Note velocities.
    """
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)

def save_pianoroll(path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    Saves onsets and frames as a pianoroll image.

    Args:
        path (str): Path to save the image.
        onsets (torch.Tensor): Onset predictions.
        frames (torch.Tensor): Frame predictions.
        onset_threshold (float, optional): Onset activation threshold.
        frame_threshold (float, optional): Frame activation threshold.
        zoom (int, optional): Image zoom factor.
    """
    onsets = (1 - (onsets.t() > onset_threshold).to(torch.uint8)).cpu()
    frames = (1 - (frames.t() > frame_threshold).to(torch.uint8)).cpu()
    both = (1 - (1 - onsets) * (1 - frames))
    image = torch.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
    image = Image.fromarray(image, 'RGB')
    image = image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)


def predictions_to_midi(midi_path, onset_pred, frame_pred, velocity_pred, onset_threshold=0.5, frame_threshold=0.5):
    """
    Converts model predictions to a MIDI file.
    
    Args:
        midi_path (str): Path to save MIDI file.
        onset_pred (torch.Tensor): Onset predictions.
        frame_pred (torch.Tensor): Frame predictions.
        velocity_pred (torch.Tensor): Velocity predictions.
        onset_threshold (float, optional): Onset activation threshold.
        frame_threshold (float, optional): Frame activation threshold.
    """
    p_est, i_est, v_est = extract_notes(onset_pred, frame_pred, velocity_pred, onset_threshold, frame_threshold)
    scaling = HOP_LENGTH / SAMPLE_RATE

    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])
    save_midi(midi_path, p_est, i_est, v_est)