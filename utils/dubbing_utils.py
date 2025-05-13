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
    language: Optional[str] = None,
    max_segment_duration: float = 10.0,  # Maximum duration for any segment to prevent stretching
    keep_temp_files: bool = False  # Option to keep temporary segment files
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
        max_segment_duration: Maximum duration for any segment
        keep_temp_files: Whether to keep temporary segment files
        
    Returns:
        Path to the dubbed audio file
    """
    print(f"Creating dubbed audio with {len(segments)} segments")
    print(f"Total audio duration target: {total_duration:.2f} seconds")
    
    # Create an empty audio array of the right length
    audio = np.zeros(int(total_duration * sample_rate))
    
    # Create a segments directory if keeping temp files
    segments_dir = os.path.join(temp_dir, "segments") if keep_temp_files else temp_dir
    if keep_temp_files:
        os.makedirs(segments_dir, exist_ok=True)
        print(f"Saving individual segment files to: {segments_dir}")
    
    # Track segment files for cleanup
    segment_files = []
    
    # Process each segment
    for i, seg in enumerate(segments):
        print(f"Processing segment {i+1}/{len(segments)}: {seg['text'][:30]}{'...' if len(seg['text']) > 30 else ''}")
        
        # Create path for this segment's audio
        seg_path = os.path.join(segments_dir, f"seg_{seg['id']}.wav")
        segment_files.append(seg_path)
        
        # Synthesize speech for this segment
        synthesize_speech(tts, seg['text'], seg_path, speaker=speaker, language=language)
        
        # Load the synthesized audio
        wav, sr = sf.read(seg_path)
        
        # If stereo, convert to mono
        if len(wav.shape) > 1 and wav.shape[1] > 1:
            wav = np.mean(wav, axis=1)
        
        # Calculate segment duration and original duration
        synth_duration = len(wav) / sr
        orig_duration = seg['end'] - seg['start']
        
        # Limit maximum segment duration to prevent output being too long
        orig_duration = min(orig_duration, max_segment_duration)
        
        print(f"  Segment {i+1} timing - Original: {orig_duration:.2f}s, Synthesized: {synth_duration:.2f}s")
        
        # Determine if we need to adjust the timing
        if abs(synth_duration - orig_duration) > 0.3:  # If difference is significant (300ms)
            print(f"  Adjusting timing for segment {i+1}")
            
            # Calculate stretch rate
            rate = synth_duration / orig_duration if orig_duration > 0 else 1.0
            
            # Don't stretch too much in either direction
            if rate > 1.5:
                rate = 1.5
            elif rate < 0.5:
                rate = 0.5
                
            # Apply time stretching
            target_len = int(orig_duration * sr)
            if target_len > 0 and len(wav) > 0:
                try:
                    wav = librosa.effects.time_stretch(wav, rate=rate)
                    print(f"    Adjusted duration: {len(wav)/sr:.2f}s (rate: {rate:.2f})")
                    
                    # Save the adjusted segment if keeping temp files
                    if keep_temp_files:
                        adjusted_path = os.path.join(segments_dir, f"seg_{seg['id']}_adjusted.wav")
                        sf.write(adjusted_path, wav, sr)
                        segment_files.append(adjusted_path)
                except Exception as e:
                    print(f"    Error adjusting timing: {e}")
        
        # Calculate the start and end positions in the output array
        start = int(seg['start'] * sample_rate)
        
        # Ensure start is within bounds
        if start >= len(audio):
            print(f"  Warning: Segment {i+1} starts beyond audio duration, skipping")
            continue
            
        # Calculate end position
        end = min(start + len(wav), len(audio))
        
        # Add the segment to the output audio with a small fade in/out
        fade_samples = min(int(0.05 * sr), len(wav) // 4)  # 50ms fade or 1/4 of segment
        
        # Apply fade in/out to segment
        if len(wav) > 2 * fade_samples:
            # Create fade in/out curves
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            # Apply fades
            wav[:fade_samples] *= fade_in
            wav[-fade_samples:] *= fade_out
        
        # Add to output audio
        audio[start:end] += wav[:end-start]
    
    # Normalize the audio to prevent clipping but maintain dynamics
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        # Use a softer normalization to preserve dynamics
        audio = audio / max_val * 0.9
    
    # Apply a subtle low-pass filter to smooth transitions
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 0.95, 'lowpass')
        audio = filtfilt(b, a, audio)
    except Exception as e:
        print(f"Warning: Could not apply smoothing filter: {e}")
    
    # Save the complete dubbed audio
    out_path = os.path.join(temp_dir, "dubbed_speech.wav")
    sf.write(out_path, audio, sample_rate)
    
    # Clean up segment files if not keeping them
    if not keep_temp_files:
        for file_path in segment_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file_path}: {e}")
    
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
    target_language: str = "en",
    keep_temp_files: bool = False,
    temp_dir_path: Optional[str] = None
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
    
    # Use provided temp directory or create a temporary one
    if keep_temp_files and temp_dir_path:
        temp_dir = temp_dir_path
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Saving temporary files to: {temp_dir}")
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory()
        temp_dir = temp_context.name
        print(f"Using temporary directory: {temp_dir}")
    
    try:
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
            language=target_language,
            keep_temp_files=keep_temp_files  # Pass the keep_temp_files parameter
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
        # Pass the music_path directly to ensure it's used in the merge process
        output_video = merge_audio_with_video(
            video_path, 
            final_audio_path, 
            output_path,
            music_path=music_path,  # Pass the original music path
            speech_volume=speech_volume,
            music_volume=music_volume
        )
        
        print(f"Dubbed video saved to {output_video}")
        return output_video
    finally:
        # Clean up temporary directory if we created one and don't want to keep files
        if temp_context and not keep_temp_files:
            temp_context.cleanup()
