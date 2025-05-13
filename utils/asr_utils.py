"""
ASR (Automatic Speech Recognition) utility functions for video dubbing tool.
Handles transcription of audio using Whisper.
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
    Transcribe audio using Whisper and return segments with timestamps.
    
    Args:
        audio_path: Path to the audio file to transcribe
        model_size: Whisper model size (tiny, base, small, medium, large)
        task: Task to perform (transcribe or translate)
        language: Source language code (e.g., 'en', 'ja', 'ar') or None for auto-detection
        
    Returns:
        List of segments with text and timing information
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
