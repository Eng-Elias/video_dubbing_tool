# Video Dubbing Tool (Streamlit GUI)

A fully-featured, multi-language video dubbing tool with a modern Streamlit GUI. Supports English, Arabic, and Japanese. Built for maintainability and speed with Whisper and Coqui TTS.

## Features
- Upload and dub videos between English, Arabic, and Japanese
- Preserves background music and timing
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

## Notes
- Requires Python 3.9+
- Make sure [ffmpeg](https://ffmpeg.org/) is installed and on your PATH.
- For best performance, use a machine with a CUDA-capable GPU.

## Project Structure
- `app.py`: Streamlit GUI
- `lib/`: Modular utility code for audio, video, dubbing, and model management
- `requirements.txt`: Main dependencies
- `requirements.lock`: (Optional) Locked, reproducible environment

## License
MIT
