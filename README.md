# ASR Data Collection Tool

A lightweight Python GUI tool for recording voice datasets specifically designed for Automatic Speech Recognition (ASR) models (e.g., Whisper). It manages phrase prompts, records audio, prevents duplicates, and automatically generates a training manifest.

## Features

- **Smart Resume**: Automatically detects existing `.wav` files and skips already recorded phrases.
- **Manifest Generation**: Creates a `dataset_manifest.jsonl` file compatible with standard ASR training pipelines.
- **Keyboard Shortcuts**: Optimized for rapid data entry without mouse usage.
- **Flexible Input**: Accepts `phrases.csv` with either single-column text or `id,text` format.
