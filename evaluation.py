from mamba_amt.inference.midi_utils import extract_notes, notes_to_frames, midi_to_hz
from mamba_amt.data.constants import *
from mamba_amt.data import MAESTRO
from mamba_amt.models import Mamba_AMT

from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from scipy.stats import hmean

from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import sys
import argparse
from tqdm import tqdm
import json


def evaluate(reference, predictions, onset_threshold=0.5, frame_threshold=0.5):
    metrics = dict()
    for key, value in predictions.items():
        value.relu_()

    p_ref, i_ref, v_ref = extract_notes(reference['onset'], reference['frame'], reference['velocity'])
    p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'], onset_threshold, frame_threshold)

    t_ref, f_ref = notes_to_frames(p_ref, i_ref, reference['frame'].shape)
    t_est, f_est = notes_to_frames(p_est, i_est, predictions['frame'].shape)

    scaling = HOP_LENGTH / SAMPLE_RATE

    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = midi_to_hz(MIN_MIDI + np.array(p_ref))
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = midi_to_hz(MIN_MIDI + np.array(p_est))

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [midi_to_hz(MIN_MIDI + np.array(freqs)) for freqs in f_ref]
    t_est = t_est.astype(np.float64) * scaling
    f_est = [midi_to_hz(MIN_MIDI + np.array(freqs)) for freqs in f_est]

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics['metric/note/precision'] = p
    metrics['metric/note/recall'] = r
    metrics['metric/note/f1'] = f
    metrics['metric/note/overlap'] = o

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics['metric/note-with-offsets/precision'] = p
    metrics['metric/note-with-offsets/recall'] = r
    metrics['metric/note-with-offsets/f1'] = f
    metrics['metric/note-with-offsets/overlap'] = o

    p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                offset_ratio=None, velocity_tolerance=0.1)
    metrics['metric/note-with-velocity/precision'] = p
    metrics['metric/note-with-velocity/recall'] = r
    metrics['metric/note-with-velocity/f1'] = f
    metrics['metric/note-with-velocity/overlap'] = o

    p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
    metrics['metric/note-with-offsets-and-velocity/precision'] = p
    metrics['metric/note-with-offsets-and-velocity/recall'] = r
    metrics['metric/note-with-offsets-and-velocity/f1'] = f
    metrics['metric/note-with-offsets-and-velocity/overlap'] = o

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    eps = sys.float_info.epsilon
    metrics['metric/frame/f1'] = hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps

    for key, loss in frame_metrics.items():
        metrics['metric/frame/' + key.lower().replace(' ', '_')] = loss

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--dataset_path', default='datasets/maestro-v3.0.0')
    parser.add_argument('-g', '--groups', nargs='?')
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    args = parser.parse_args()

    config = json.load(open("mamba_amt/configs/mamba_amt.json", "r"))
    model_config = config['model']
    training_config = config['training']

    dataset = MAESTRO(path=args.dataset_path, sequence_length=training_config["sequence_length"], groups=[args.groups])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Mamba_AMT.load_from_checkpoint(args.ckpt_path, model_config=model_config, lr=1e-3)
    all_metrics = defaultdict(list)
    for batch in tqdm(loader):
        with torch.no_grad():
            predictions, loss = model.run_on_batch(batch)
        metrics = evaluate(batch, predictions, args.onset_threshold, args.frame_threshold)

        for key, value in metrics.items():
            all_metrics[key].append(metrics[key])
    
    for key, value in all_metrics.items():
        print(f"{key}: {np.mean(value)}")