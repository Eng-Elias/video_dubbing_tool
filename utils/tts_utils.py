"""
TTS (Text-to-Speech) utility functions for video dubbing tool.

This module provides functions for text-to-speech synthesis using Coqui TTS models,
including support for voice cloning. It handles model initialization, speech synthesis,
and provides utility functions for listing available speakers and languages.

Key functions:
- initialize_tts: Initialize a TTS model with optional voice cloning support
- synthesize_speech: Generate speech from text with optional voice cloning
- list_speakers_for_model: Get available speakers for a specific model
- list_languages_for_model: Get available languages for a specific model
- get_voice_cloning_models: Get information about available voice cloning models
"""
import torch
import importlib
from TTS.api import TTS
import soundfile as sf
from typing import Optional, Tuple, Any, List, Dict

def initialize_tts(
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", 
    gpu: bool = True,
    reference_wav: Optional[str] = None,
    reference_speaker_lang: Optional[str] = None
) -> Any:
    """
    Initialize Coqui TTS model with optional voice cloning support.
    
    This function initializes a Text-to-Speech model from the Coqui TTS library,
    configuring it based on the specified parameters. It handles device selection
    (CPU/GPU) and provides special handling for XTTS models that may have compatibility
    issues with PyTorch 2.6+ due to the weights_only parameter change.
    
    Algorithm:
    1. Check if GPU is available and should be used
    2. Determine if voice cloning is requested based on model name and reference audio
    3. For XTTS models, apply special handling to address PyTorch compatibility issues:
       a. Try to add XttsConfig to torch safe globals
       b. If that fails, temporarily patch torch.load to use weights_only=False
    4. Initialize and return the TTS model
    
    Args:
        model_name: Name of the TTS model to use (e.g., "tts_models/en/ljspeech/tacotron2-DDC")
        gpu: Whether to use GPU acceleration if available
        reference_wav: Path to reference audio file for voice cloning
        reference_speaker_lang: Language of the reference speaker for voice cloning
        
    Returns:
        Initialized TTS model ready for speech synthesis
    """
    print(f"Initializing TTS model: {model_name}")
    use_gpu = gpu and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    
    # Check if we're using a voice cloning model
    is_voice_cloning = False
    if reference_wav and ('your_tts' in model_name or 'xtts' in model_name):
        is_voice_cloning = True
        print(f"Voice cloning enabled with reference audio: {reference_wav}")
        if reference_speaker_lang:
            print(f"Reference speaker language: {reference_speaker_lang}")
    
    # Handle XTTS model specifically due to PyTorch 2.6 weights_only issue
    if 'xtts' in model_name:
        try:
            # Try to add the safe globals for XTTS
            try:
                # Check if we have the serialization module with add_safe_globals
                if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                    # Import the XttsConfig class
                    try:
                        from TTS.tts.configs.xtts_config import XttsConfig
                        # Add it to safe globals
                        torch.serialization.add_safe_globals([XttsConfig])
                        print("Added XttsConfig to torch safe globals")
                    except ImportError:
                        print("Could not import XttsConfig, trying to load model anyway")
            except Exception as e:
                print(f"Warning: Could not set up safe globals: {e}")
                
            # Initialize the TTS model
            tts = TTS(model_name, gpu=use_gpu)
            print(f"XTTS model loaded on {device}")
        except Exception as e:
            print(f"Error loading XTTS model with weights_only=True: {e}")
            print("Trying to load with weights_only=False (this is safe if you trust the model)")
            
            # Monkey patch torch.load to use weights_only=False for this call
            original_torch_load = torch.load
            
            def patched_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            # Replace torch.load temporarily
            torch.load = patched_torch_load
            
            try:
                # Try loading with the patched function
                tts = TTS(model_name, gpu=use_gpu)
                print(f"XTTS model loaded with weights_only=False on {device}")
            except Exception as e2:
                print(f"Failed to load XTTS model: {e2}")
                raise e2
            finally:
                # Restore original torch.load
                torch.load = original_torch_load
    else:
        # Initialize the TTS model for non-XTTS models
        tts = TTS(model_name, gpu=use_gpu)
        print(f"TTS model loaded on {device}")
    
    return tts

def synthesize_speech(
    tts: Any, 
    text: str, 
    output_path: str, 
    speaker: Optional[str] = None, 
    language: Optional[str] = None,
    reference_wav: Optional[str] = None,
    reference_speaker_name: Optional[str] = None
) -> Tuple[Any, int]:
    """
    Synthesize speech using Coqui TTS with optional voice cloning support.
    
    This function generates speech from text using a Coqui TTS model, handling various
    model types including voice cloning models. It automatically detects if voice cloning
    should be used based on the model name and reference audio availability.
    
    Algorithm:
    1. Check if we're using a voice cloning model (YourTTS or XTTS) with reference audio
    2. If using voice cloning:
       a. Generate speech using the reference audio as the voice source
    3. If not using voice cloning:
       a. Prepare appropriate parameters based on model capabilities (speaker, language)
       b. Generate speech using standard TTS
    4. Load and return the generated audio file
    
    Args:
        tts: Initialized TTS model
        text: Text to synthesize
        output_path: Path to save the synthesized audio
        speaker: Speaker ID for multi-speaker models
        language: Language code for multi-language models
        reference_wav: Path to reference audio file for voice cloning
        reference_speaker_name: Name to identify the reference speaker
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    print(f"Synthesizing speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Check if we're using a voice cloning model
    is_voice_cloning = False
    model_name = tts.model_name if hasattr(tts, 'model_name') else str(tts)
    
    if reference_wav and ('your_tts' in model_name or 'xtts' in model_name):
        is_voice_cloning = True
        print(f"Using voice cloning with reference audio: {reference_wav}")
        
        # Generate speech with voice cloning
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=reference_wav,
            language=language
        )
    else:
        # Standard TTS without voice cloning
        kwargs = {"text": text, "file_path": output_path}
        
        # Add speaker if available and model supports it
        if speaker and hasattr(tts, 'speakers') and speaker in tts.speakers:
            kwargs["speaker"] = speaker
            
        # Add language if available and model supports it
        if language and hasattr(tts, 'languages') and language in tts.languages:
            kwargs["language"] = language
            
        # Generate speech
        tts.tts_to_file(**kwargs)
    
    # Load the generated audio to return
    audio, sr = sf.read(output_path)
    
    print(f"Speech synthesized and saved to: {output_path}")
    return audio, sr

def list_speakers_for_model(model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> List[str]:
    """
    List available speakers for a specific TTS model.
    
    This function initializes a TTS model and retrieves the list of available speakers
    if the model supports multiple speakers. For single-speaker models, it returns an
    empty list.
    
    Algorithm:
    1. Initialize the specified TTS model
    2. Check if the model has a 'speakers' attribute
    3. Return the list of speakers if available, otherwise return an empty list
    
    Args:
        model_name: Name of the TTS model to check for available speakers
        
    Returns:
        List of available speaker IDs as strings
    """
    # Initialize the model
    tts = TTS(model_name)
    
    # Get the list of speakers if available
    if hasattr(tts, 'speakers'):
        return tts.speakers
    else:
        return []

def list_languages_for_model(model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> List[str]:
    """
    List available languages for a specific TTS model.
    
    This function initializes a TTS model and retrieves the list of available languages
    if the model supports multiple languages. For single-language models, it returns an
    empty list.
    
    Algorithm:
    1. Initialize the specified TTS model
    2. Check if the model has a 'languages' attribute
    3. Return the list of languages if available, otherwise return an empty list
    
    Args:
        model_name: Name of the TTS model to check for available languages
        
    Returns:
        List of available language codes as strings
    """
    # Initialize the model
    tts = TTS(model_name)
    
    # Get the list of languages if available
    if hasattr(tts, 'languages'):
        return tts.languages
    else:
        return []


def get_voice_cloning_models() -> List[Dict[str, str]]:
    """
    Get a list of available voice cloning models.
    
    This function returns information about voice cloning models available in Coqui TTS.
    Each model is represented as a dictionary with id, name, and description fields.
    Currently, it returns information about YourTTS and XTTS v2 models.
    
    Returns:
        List of dictionaries containing model information with the following keys:
        - id: Model identifier used for loading the model
        - name: Human-readable name of the model
        - description: Brief description of the model's capabilities
    """
    return [
        {
            "id": "tts_models/multilingual/multi-dataset/your_tts",
            "name": "YourTTS",
            "description": "Multilingual voice cloning model with high quality"
        },
        {
            "id": "tts_models/multilingual/multi-dataset/xtts_v2",
            "name": "XTTS v2",
            "description": "Advanced voice cloning model with improved quality"
        }
    ]
