# Video Dubbing Tool (Streamlit GUI)

A fully-featured, multi-language video dubbing tool with a modern Streamlit GUI. Supports English, Arabic, and Japanese. Built for maintainability and speed with Whisper and Coqui TTS.

## Features

- Upload and dub videos between English, Arabic, and Japanese
- Enhanced background music preservation with dynamic volume control
- Adjustable speech and music volume controls
- Option to save temporary processing files for review
- Advanced audio processing with noise reduction and speech clarity enhancement
- Precise audio-video synchronization
- GPU acceleration (if available)
- Modular, clean codebase
- Fast dependency management with [uv](https://github.com/astral-sh/uv)

## Quick Start

### 1. Create and Activate a Python Virtual Environment

```sh
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install [uv](https://github.com/astral-sh/uv)

```sh
pip install uv
```

### 3. Install Project Dependencies

```sh
uv pip install -r requirements.txt
```

### 4. (Recommended) Generate a Lockfile

```sh
uv pip compile requirements.txt --output requirements.lock
```

### 5. (Optional) Sync Environment from Lockfile

```sh
uv pip sync requirements.lock
```

### 6. Run the App

```sh
streamlit run app.py
```

The app will open in your browser. Upload a video and start dubbing!

## Usage and Advanced Features

### Audio Controls

- **Speech Volume**: Adjust the volume of the synthesized speech (0.0-2.0, default: 1.5)
- **Music Volume**: Adjust the volume of the background music (0.0-2.0, default: 1.0)
- **Preserve Background Music**: Toggle to extract and preserve music from the original video
- **Apply Noise Reduction**: Toggle to apply noise reduction to the original audio for better transcription

### Temporary Files

- **Save Temporary Files**: Enable this option to save all intermediate processing files
- When enabled, files are saved in the `output` directory with a timestamp
- Useful for debugging or reviewing individual processing steps
- Files include: extracted audio, music tracks, speech segments, and mixed audio

### Dynamic Music Mixing

- The tool automatically adjusts music volume during silent speech segments
- This creates a more natural listening experience where music becomes more prominent when no one is speaking
- Speech clarity is enhanced through subtle EQ processing

## Notes

- Requires Python 3.9+
- Make sure [ffmpeg](https://ffmpeg.org/) is installed and on your PATH.
- For best performance, use a machine with a CUDA-capable GPU.

## Project Structure

- `app.py`: Streamlit GUI
- `utils/`: Modular utility code for audio, video, dubbing, and model management
  - `audio_utils.py`: Audio processing functions (extraction, mixing, noise reduction)
  - `video_utils.py`: Video processing functions
  - `dubbing_utils.py`: Main dubbing pipeline
  - `asr_utils.py`: Automatic speech recognition (Whisper)
  - `tts_utils.py`: Text-to-speech synthesis (Coqui TTS)
- `output/`: Directory for storing temporary processing files (when enabled)
- `requirements.txt`: Main dependencies
- `requirements.lock`: (Optional) Locked, reproducible environment

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
