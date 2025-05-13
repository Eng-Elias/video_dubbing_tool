"""
TTS (Text-to-Speech) utility functions for video dubbing tool.
Handles speech synthesis using Coqui TTS.
"""
import torch
from TTS.api import TTS
import soundfile as sf
from typing import Optional, Tuple, Any

def initialize_tts(model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", gpu: bool = True) -> Any:
    """
    Initialize Coqui TTS model.
    
    Args:
        model_name: Name of the TTS model to use
        gpu: Whether to use GPU acceleration if available
        
    Returns:
        Initialized TTS model
    """
    print(f"Initializing TTS model: {model_name}")
    use_gpu = gpu and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    
    # Initialize the TTS model
    tts = TTS(model_name, gpu=use_gpu)
    print(f"TTS model loaded on {device}")
    
    return tts

def synthesize_speech(
    tts: Any, 
    text: str, 
    output_path: str, 
    speaker: Optional[str] = None, 
    language: Optional[str] = None
) -> Tuple[Any, int]:
    """
    Synthesize speech using Coqui TTS.
    
    Args:
        tts: Initialized TTS model
        text: Text to synthesize
        output_path: Path to save the synthesized audio
        speaker: Speaker ID for multi-speaker models
        language: Language code for multi-language models
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    print(f"Synthesizing speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Generate speech
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker=speaker,
        # language=language
    )
    
    # Load the generated audio to return
    audio, sr = sf.read(output_path)
    
    print(f"Speech synthesized and saved to: {output_path}")
    return audio, sr

def list_speakers_for_model(model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> list:
    """
    List available speakers for a specific model.
    
    Args:
        model_name: Name of the TTS model
        
    Returns:
        List of available speaker IDs
    """
    # Initialize the model
    tts = TTS(model_name)
    
    # Get the list of speakers if available
    if hasattr(tts, 'speakers'):
        return tts.speakers
    else:
        return []

def list_languages_for_model(model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> list:
    """
    List available languages for a specific model.
    
    Args:
        model_name: Name of the TTS model
        
    Returns:
        List of available language codes
    """
    # Initialize the model
    tts = TTS(model_name)
    
    # Get the list of languages if available
    if hasattr(tts, 'languages'):
        return tts.languages
    else:
        return []
