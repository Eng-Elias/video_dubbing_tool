"""
Dubbing utility functions for video dubbing tool.
Handles the complete dubbing pipeline and workflow.
"""
import os
import numpy as np
import soundfile as sf
import tempfile
from typing import List, Dict, Any, Optional
from moviepy.editor import VideoFileClip

from utils.audio_utils import extract_audio, apply_noise_reduction, extract_music, mix_audio_tracks
from utils.asr_utils import transcribe_audio
from utils.tts_utils import initialize_tts, synthesize_speech
from utils.video_utils import merge_audio_with_video

def create_dubbed_audio(
    segments: List[Dict[str, Any]],
    tts: Any,
    total_duration: float,
    sample_rate: int,
    temp_dir: str,
    speaker: Optional[str] = None,
    language: Optional[str] = None
) -> str:
    """
    Create dubbed audio from segments using Coqui TTS.
    
    Args:
        segments: List of transcribed segments with timing information
        tts: Initialized TTS model
        total_duration: Total duration of the audio in seconds
        sample_rate: Sample rate for the output audio
        temp_dir: Directory to store temporary files
        speaker: Speaker ID for multi-speaker models
        language: Language code for multi-language models
        
    Returns:
        Path to the dubbed audio file
    """
    print(f"Creating dubbed audio with {len(segments)} segments")
    
    # Create an empty audio array of the right length
    audio = np.zeros(int(total_duration * sample_rate))
    
    # Process each segment
    for i, seg in enumerate(segments):
        print(f"Processing segment {i+1}/{len(segments)}: {seg['text'][:30]}{'...' if len(seg['text']) > 30 else ''}")
        
        # Create path for this segment's audio
        seg_path = os.path.join(temp_dir, f"seg_{seg['id']}.wav")
        
        # Synthesize speech for this segment
        synthesize_speech(tts, seg['text'], seg_path, speaker=speaker, language=language)
        
        # Load the synthesized audio
        wav, sr = sf.read(seg_path)
        
        # If stereo, convert to mono
        if len(wav.shape) > 1 and wav.shape[1] > 1:
            wav = np.mean(wav, axis=1)
        
        # Calculate the start and end positions in the output array
        start = int(seg['start'] * sample_rate)
        end = min(start + len(wav), len(audio))
        
        # Add the segment to the output audio
        audio[start:end] += wav[:end-start]
    
    # Normalize the audio to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9
    
    # Save the complete dubbed audio
    out_path = os.path.join(temp_dir, "dubbed_speech.wav")
    sf.write(out_path, audio, sample_rate)
    
    print(f"Complete dubbed audio saved to: {out_path}")
    return out_path

def process_video(
    video_path: str,
    output_path: str,
    source_language: Optional[str] = None,
    preserve_music: bool = True,
    apply_noise_red: bool = True,
    speech_volume: float = 1.0,
    music_volume: float = 0.3,
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC",
    speaker: Optional[str] = None,
    target_language: str = "en"
) -> str:
    """
    Process a video through the complete dubbing pipeline.
    
    Args:
        video_path: Path to the input video
        output_path: Path for the output dubbed video
        source_language: Source language code or None for auto-detection
        preserve_music: Whether to extract and preserve background music
        apply_noise_red: Whether to apply noise reduction
        speech_volume: Volume multiplier for speech
        music_volume: Volume multiplier for music
        tts_model: TTS model to use
        speaker: Speaker ID for multi-speaker models
        target_language: Target language code
        
    Returns:
        Path to the output dubbed video
    """
    print(f"Starting video dubbing process for: {video_path}")
    print(f"Source language: {source_language or 'auto-detect'}, Target language: {target_language}")
    
    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Extract audio from video
        extracted_audio = extract_audio(video_path, os.path.join(temp_dir, "extracted_audio.wav"))
        
        # Step 2: Extract music track if requested
        music_path = None
        if preserve_music:
            try:
                music_path = extract_music(extracted_audio, os.path.join(temp_dir, "music.wav"))
                print(f"Music extraction {'succeeded' if music_path else 'failed'}")
            except Exception as e:
                print(f"Music extraction error: {e}")
                music_path = None
        
        # Step 3: Apply noise reduction for better transcription if requested
        transcription_audio = extracted_audio
        if apply_noise_red:
            try:
                cleaned_audio = apply_noise_reduction(extracted_audio, os.path.join(temp_dir, "cleaned_audio.wav"))
                transcription_audio = cleaned_audio
            except Exception as e:
                print(f"Noise reduction error: {e}")
        
        # Step 4: Transcribe audio
        task = "translate" if source_language != target_language else "transcribe"
        segments = transcribe_audio(
            transcription_audio, 
            model_size="medium",  # Always use medium model for better accuracy
            task=task,
            language=source_language
        )
        
        # Step 5: Get video duration for full audio synthesis
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()
        
        # Step 6: Initialize TTS
        tts = initialize_tts(tts_model, gpu=True)
        
        # Step 7: Get sample rate from a test synthesis
        test_path = os.path.join(temp_dir, "test.wav")
        _, sample_rate = synthesize_speech(
            tts, 
            "Test", 
            test_path, 
            speaker=speaker, 
            language=target_language
        )
        
        # Step 8: Create dubbed audio
        dubbed_speech = create_dubbed_audio(
            segments, 
            tts, 
            duration, 
            sample_rate, 
            temp_dir,
            speaker=speaker,
            language=target_language
        )
        
        # Step 9: Mix speech with music if available
        if music_path:
            mixed_audio_path = os.path.join(temp_dir, "mixed_audio.wav")
            mix_audio_tracks(
                dubbed_speech,
                music_path,
                mixed_audio_path,
                speech_volume=speech_volume,
                music_volume=music_volume
            )
            final_audio_path = mixed_audio_path
        else:
            final_audio_path = dubbed_speech
        
        # Step 10: Merge dubbed audio with original video
        output_video = merge_audio_with_video(
            video_path, 
            final_audio_path, 
            output_path,
            music_path=None,  # Music is already mixed in
            speech_volume=1.0,  # Already applied in mixing
            music_volume=0.0    # Already applied in mixing
        )
        
        print(f"Dubbed video saved to {output_video}")
        return output_video
