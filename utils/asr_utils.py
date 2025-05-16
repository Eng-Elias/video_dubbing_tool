"""
ASR (Automatic Speech Recognition) utility functions for video dubbing tool.

This module provides functions for transcribing and translating audio using OpenAI's
Whisper model. It handles the loading of appropriate Whisper models, device selection
(CPU/GPU), and processing of audio files to extract text with timing information.

Key functions:
- transcribe_audio: Transcribe or translate audio using Whisper and return segments with timestamps
"""
import whisper
import torch
from typing import List, Dict, Any, Optional

def transcribe_audio(
    audio_path: str, 
    model_size: str = "medium", 
    task: str = "transcribe", 
    language: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Transcribe audio using OpenAI's Whisper model and return segments with timestamps.
    
    This function loads the specified Whisper model and processes the audio file to generate
    a transcription or translation with detailed segment information including timestamps.
    It automatically uses GPU acceleration if available for faster processing.
    
    Algorithm:
    1. Load the specified Whisper model size (tiny, base, small, medium, large)
    2. Use GPU if available, otherwise fall back to CPU
    3. Process the audio file with the model to perform transcription or translation
    4. Extract and return the segments with text and timing information
    
    Args:
        audio_path: Path to the audio file to transcribe
        model_size: Whisper model size (tiny, base, small, medium, large)
        task: Task to perform (transcribe or translate)
        language: Source language code (e.g., 'en', 'ja', 'ar') or None for auto-detection
        
    Returns:
        List of segment dictionaries containing text, start and end timestamps, etc.
    """
    print(f"Loading Whisper {model_size} model...")
    # Use GPU if available for faster processing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    
    print(f"Transcribing audio: {audio_path}")
    print(f"Task: {task}, Language: {language or 'auto-detect'}")
    
    # Perform transcription/translation
    result = model.transcribe(
        audio_path,
        task=task,
        language=language,
        verbose=False
    )
    
    print(f"Transcription complete: {len(result['segments'])} segments")
    return result["segments"]
