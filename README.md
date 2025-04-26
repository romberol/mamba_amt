# Exploring Mamba Architectures for Piano Transcription

This repository is a fork and adaptation of [State Spaces' Mamba](https://github.com/state-spaces/mamba/), designed for the task of Automatic Music Transcription (AMT), specifically for piano transcription.

It leverages the efficient Mamba architecture to transcribe raw audio into piano rolls and MIDI files.

## Installation

```bash
pip install .
```

Try passing `--no-build-isolation` to `pip` if installation encounters difficulties.  
For Windows users, its recommended to use WSL2 for installation and usage.

Other requirements:
- Linux
- NVIDIA GPU
- CUDA 11.6+

For AMD cards, see additional prerequisites below.


## Inference
Transcribe a piano audio file into a MIDI:
```bash
python inference.py checkpoint.pt -a path/to/audio.wav -o path/to/output_dir
```
Arguments:
- `checkpoint.pt`: Path to a trained model checkpoint.
- `-a / --audio`: Path to input audio.
- `-o / --output`: Output directory to save generated .mid and pianoroll plots. (default: ./output)
- `--onset_threshold`: Onset detection probability threshold.
- `--frame_threshold`: Frame activation probability threshold.

Example:
```bash
python inference.py my_model.pt -a example.wav -o results/
```
Outputs:

- `results/transcription.midi`: MIDI file.
- `results/pianoroll.png`: Visualization of the transcription.


## Training
Train the model on the MAESTRO dataset:

```bash
python train.py
```
Training configurations are specified in `mamba_amt/configs/mamba_amt.json`:
- Model architecture (number of Mamba blocks, hidden size, etc.)
- Training hyperparameters (batch size, learning rate schedule, number of epochs, etc.)
- Dataset settings (paths to MAESTRO dataset, noise and impulse response files for augmentation).

Training uses WandB for experiment tracking. Make sure to login:
```bash
wandb login
```

## Pretrained Model

The weights of the model, for which evaluation metrics are reported in the thesis, are available for download:

[Download pretrained checkpoint](https://drive.google.com/file/d/1LRR_xmMfV8zvUSjKA701yJIZJLCI8Nq6/view?usp=sharing)

You can use this checkpoint for inference or further evaluation.

## Evaluation
Evaluate a trained model:

```bash
python evaluation.py checkpoint.pt --dataset_path path/to/maestro-v3.0.0 -g test
```
Arguments:
- `checkpoint.pt`: Path to model checkpoint.
- `--dataset_path`: Path to MAESTRO dataset.
- `-g / --groups`: Evaluation group (`train`, `val`, `test`, or specific year).
- `--onset-threshold`: Onset threshold. (default: `0.5`)
- `--frame-threshold`: Frame threshold. (default: `0.5`)
- `-f / --full-tracks`: Evaluate on full-length tracks instead of fixed sequence segments.
Example:
```bash
python evaluation.py my_model.pt --dataset_path ./datasets/maestro-v3.0.0 -g test -f
```

## Configuration
The training, model, and dataset settings are controlled via the JSON file `mamba_amt/configs/mamba_amt.json`.
### Model Configuration
The model expects the following structure for its configuration:
- `mamba_config` (**required**):  
  A dictionary that defines the internal parameters of the Mamba layer. Example keys:
  - `d_model`: Model dimension.
  - `d_state`: SSM state expansion factor.
  - `d_conv`: Local convolution width.
  - `expand`: Expansion factor for intermediate channels.

- `bidirectional_cfg` (**optional**):  
  A dictionary that controls **bidirectional** Mamba processing. Possible keys:
  - `shared` (bool):  
    If `True`, forward and backward Mamba layers will **share the same weights**.  
    If `False`, they will have **separate parameters**.
  - `concat` (bool):  
    If `True`, outputs from forward and backward passes are **concatenated** along the feature dimension.  
    If `False`, they are **summed** together.

If `bidirectional_cfg` is set to `null`, the model will behave as a **standard unidirectional Mamba**.

Example:
```json
"bidirectional_cfg": {
  "shared": true,
  "concat": false
}
```

### Dataset Configuration
The dataset settings are defined under the `dataset` section. Required keys:
- `path`: Path to the MAESTRO dataset directory.
- `noise_path`:  
  Path to the noise files for augmentation.  
  These files are randomly mixed into training samples for audio augmentation to improve model robustness.
- `ir_path`:   
  Path to a folder containing impulse response (IR) recordings.  
  These are used to simulate realistic reverb by convolving IRs with the piano recordings during augmentation.

Make sure all dataset paths are correctly set before training.


## Additional Prerequisites for AMD cards

### Patching ROCm

If you are on ROCm 6.0, run the following steps to avoid errors during compilation. This is not required for ROCm 6.1 onwards.

1. Locate your ROCm installation directory. This is typically found at `/opt/rocm/`, but may vary depending on your installation.

2. Apply the Patch. Run with `sudo` in case you encounter permission issues.
   ```bash
    patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h < rocm_patch/rocm6_0.patch 
   ```
