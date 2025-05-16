"""
Audio utility functions for video dubbing tool.

This module provides functions for audio processing tasks required for video dubbing,
including audio extraction, noise reduction, voice sample extraction for cloning,
music extraction, and audio mixing.

Key functions:
- extract_audio: Extract audio from a video file
- extract_voice_sample: Extract a clean voice sample for voice cloning
- apply_noise_reduction: Reduce background noise in audio
- extract_music: Extract music track while removing vocals
- mix_audio_tracks: Mix speech and music tracks with volume control

Additional utility functions for audio processing are also provided.
"""
import os
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from moviepy.editor import VideoFileClip
from typing import Optional, Tuple
from scipy.signal import butter, filtfilt
import subprocess
import torch

def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio from a video file and save it as a WAV file.
    
    This function uses MoviePy to extract the audio track from a video file and save it
    as a high-quality WAV file. It preserves stereo audio if available, which is important
    for better music separation in later processing steps.
    
    Algorithm:
    1. If no output path is provided, generate one based on the input video filename
    2. Load the video file using MoviePy's VideoFileClip
    3. Extract the audio track and save it as a PCM WAV file (uncompressed for best quality)
    4. Close the video file to free resources
    5. Return the path to the extracted audio file
    """
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_extracted.wav"
    
    print(f"Extracting audio from {video_path}...")
    video = VideoFileClip(video_path)
    # Extract stereo audio if available (for better music separation)
    video.audio.write_audiofile(output_path, codec='pcm_s16le')
    video.close()
    
    return output_path


def extract_voice_sample(audio_path: str, output_path: Optional[str] = None, duration: float = 10.0) -> str:
    """
    Extract a clean voice sample from the input audio for voice cloning purposes.
    
    This function analyzes an audio file to find segments containing clean speech,
    which are optimal for voice cloning. It applies noise reduction to improve quality
    and uses energy-based voice activity detection to identify speech segments.
    The function selects the best segments based on duration and energy level.
    
    Algorithm:
    1. If no output path is provided, generate one based on the input audio filename
    2. Load the audio file using librosa
    3. Apply noise reduction to get cleaner voice
    4. Detect speech segments using energy-based voice activity detection (VAD)
    5. Sort segments by duration (longest first)
    6. Collect multiple speech segments until reaching the target duration
    7. Concatenate the selected segments to create a voice sample
    8. Apply additional noise reduction to the final sample
    9. Save the voice sample to the output path
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path where the voice sample will be saved.
            If None, a default path will be generated based on the input filename.
        duration: Target duration in seconds for the voice sample.
            Defaults to 10.0 seconds.
    
    Returns:
        Path to the extracted voice sample file
    """
    if output_path is None:
        base_dir = os.path.dirname(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_voice_sample.wav")
    
    print(f"Extracting voice sample from {audio_path}...")
    
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Apply noise reduction to get cleaner voice
    y_reduced = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=True,
        prop_decrease=0.75
    )
    
    def detect_speech_segments(audio: np.ndarray, sr: int, frame_length: int = 1024, hop_length: int = 512, threshold: float = 0.05) -> list:
        """
        Detect speech segments in an audio signal based on energy thresholding.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of the audio
            frame_length: Length of each frame for energy calculation
            hop_length: Number of samples between frames
            threshold: Energy threshold for speech detection
            
        Returns:
            List of tuples containing (start_time, end_time) for each detected speech segment
        """
        # Calculate the energy of each frame
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Apply a threshold to detect speech frames
        speech_frames = energy > threshold
        
        # Find speech segments
        speech_segments = []
        in_speech = False
        start_frame = 0
        prev_frame = 0
        
        for i, frame in enumerate(speech_frames):
            if frame and not in_speech:
                # Start of a speech segment
                in_speech = True
                start_frame = i
            elif not frame and in_speech:
                # End of a speech segment
                in_speech = False
                # Only add segments longer than 0.5 seconds
                if (i - start_frame) * hop_length / sr > 0.5:
                    start_time = start_frame * hop_length / sr
                    end_time = i * hop_length / sr
                    speech_segments.append((start_time, end_time))
            prev_frame = i
        
        # Handle the case where the audio ends during speech
        if in_speech:
            end_time = len(audio) / sr
            start_time = start_frame * hop_length / sr
            if end_time - start_time > 0.5:
                speech_segments.append((start_time, end_time))
        
        return speech_segments
    
    # Find speech segments
    speech_segments = detect_speech_segments(y_reduced, sr)
    
    # Sort segments by duration (descending)
    speech_segments.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    # Collect segments until we reach the target duration
    selected_segments = []
    total_selected_duration = 0
    
    for start, end in speech_segments:
        seg_duration = end - start
        if total_selected_duration + seg_duration <= duration:
            selected_segments.append((start, end))
            total_selected_duration += seg_duration
        
        if total_selected_duration >= duration:
            break
    
    # If we don't have enough segments, use what we have
    if not selected_segments:
        print("No clear speech segments found. Using the beginning of the audio.")
        selected_segments = [(0, min(duration, len(y) / sr))]
    
    # Sort segments by start time
    selected_segments.sort(key=lambda x: x[0])
    
    # Concatenate selected segments
    voice_sample = np.array([])
    for start, end in selected_segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y_reduced[start_sample:end_sample]
        voice_sample = np.concatenate((voice_sample, segment))
    
    # Apply additional noise reduction to the final sample
    voice_sample = nr.reduce_noise(
        y=voice_sample,
        sr=sr,
        stationary=True,
        prop_decrease=0.8
    )
    
    # Save the voice sample
    sf.write(output_path, voice_sample, sr)
    
    print(f"Voice sample extracted and saved to {output_path} (duration: {len(voice_sample)/sr:.2f}s)")
    return output_path

def apply_noise_reduction(audio_path: str, output_path: Optional[str] = None) -> str:
    """
    Apply noise reduction to audio to improve transcription accuracy.
    
    This function processes an audio file to reduce background noise and remove
    low-frequency rumble, which significantly improves the quality of speech
    for transcription and voice cloning purposes.
    
    Algorithm:
    1. If no output path is provided, generate one based on the input filename
    2. Load the audio file using librosa
    3. Estimate noise profile from the first 2 seconds (assumed to be non-speech)
    4. Apply noise reduction using the noisereduce library
    5. Apply a high-pass filter to remove low-frequency rumble (below 80Hz)
    6. Save the processed audio to the output path
    7. Return the path to the noise-reduced audio file
    """
    if output_path is None:
        base, ext = os.path.splitext(audio_path)
        output_path = f"{base}_cleaned{ext}"
    
    print("Applying noise reduction...")
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Estimate noise profile from a presumably non-speech portion (first 2 seconds)
    noise_sample = y[:min(len(y), int(2 * sr))]
    
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=True,
        prop_decrease=0.75,
        n_std_thresh_stationary=1.5
    )
    
    # Apply a high-pass filter to remove low-frequency rumble
    b, a = butter_highpass(cutoff=80, fs=sr, order=4)
    filtered_audio = filtfilt(b, a, reduced_noise)
    
    # Save the processed audio
    sf.write(output_path, filtered_audio, sr)
    print(f"Noise-reduced audio saved to {output_path}")
    
    return output_path

def butter_highpass(cutoff: float, fs: float, order: int = 5) -> Tuple:
    """Create a highpass Butterworth filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def extract_music(audio_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract music from the audio while removing vocal content.
    
    This function attempts to separate the music track from the input audio file,
    removing vocals to create a clean instrumental track. It uses a two-tiered approach:
    1. First try to use Demucs (Facebook's state-of-the-art source separation)
    2. If Demucs fails or is not available, fall back to a spectral-based approach
    
    Algorithm:
    1. If no output path is provided, generate one based on the input filename
    2. Try to use Demucs for high-quality source separation:
       a. Load the best available Demucs model
       b. Process the audio to separate stems (vocals, drums, bass, other)
       c. Combine all stems except vocals to create a music track
       d. Apply post-processing to enhance quality
    3. If Demucs fails, fall back to spectral-based vocal removal:
       a. Apply spectral masking to reduce vocal content
       b. Apply bandpass filtering to focus on musical frequencies
       c. Apply dynamic compression to balance levels
    4. Save the extracted music to the output path
    5. Return the path to the extracted music file
    """
    if output_path is None:
        base_dir = os.path.dirname(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_music.wav")
    
    # Create a directory for Demucs output
    output_dir = os.path.join(os.path.dirname(output_path), "demucs_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting music from audio using Demucs with enhanced vocal removal...")
    try:
        # First try to use Demucs (Facebook's state-of-the-art source separation)
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torchaudio
        
        print("Loading Demucs model...")
        # Try to load the best model (htdemucs_ft)
        try:
            model = get_model("htdemucs_ft")
            print("Using htdemucs_ft model (highest quality)")
        except:
            try:
                model = get_model("htdemucs")
                print("Using htdemucs model")
            except:
                model = get_model("mdx_extra")
                print("Using mdx_extra model")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load audio
        print("Loading audio file...")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed (Demucs expects 44.1kHz)
        if sample_rate != 44100:
            print(f"Resampling from {sample_rate}Hz to 44100Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 44100)
            waveform = resampler(waveform)
            sample_rate = 44100
        
        # Convert to expected format
        waveform = waveform.to(device)
        
        # Apply the separation model
        print("Applying source separation (this may take a while)...")
        with torch.no_grad():
            sources = apply_model(model, waveform.unsqueeze(0), device=device)[0]
        
        # Sources will be in order: drums, bass, other, vocals
        # We want everything except vocals for the music track
        drums = sources[0].cpu().numpy()
        bass = sources[1].cpu().numpy()
        other = sources[2].cpu().numpy()
        
        # Combine all non-vocal sources with appropriate levels
        print("Combining instrumental tracks...")
        drums_gain = 1.0
        bass_gain = 1.2
        other_gain = 1.0
        
        # Create a balanced mix
        music = drums * drums_gain + bass * bass_gain + other * other_gain
        
        # Apply additional vocal removal using spectral masking
        music = apply_spectral_vocal_reduction(music, sample_rate)
        
        # Apply dynamic range compression to make the music more consistent
        music = apply_dynamic_compression(music, threshold=0.5, ratio=1.5)
        
        # Normalize the music to prevent clipping
        print("Normalizing audio levels...")
        if len(music.shape) > 1:
            # Stereo normalization
            max_val = np.max(np.abs(music))
            if max_val > 0:
                music = music / max_val * 0.9
        else:
            # Mono normalization
            max_val = np.max(np.abs(music))
            if max_val > 0:
                music = music / max_val * 0.9
        
        # Apply a slight boost to make music more prominent
        music = music * 1.5
        
        # Save the processed music
        print(f"Saving music track to {output_path}")
        if len(music.shape) > 1:
            # Transpose if needed for soundfile
            sf.write(output_path, music.T, sample_rate)
        else:
            sf.write(output_path, music, sample_rate)
            
        print(f"Enhanced music track extracted to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error extracting music with Demucs: {e}")
        # Fallback method: try to extract music using harmonic-percussive source separation
        try:
            print("Falling back to librosa's HPSS for music extraction...")
            y, sr = librosa.load(audio_path, sr=None)
            
            print("Performing harmonic-percussive source separation...")
            # Compute the harmonic and percussive components
            # Harmonic content often contains the musical elements we want to preserve
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Apply vocal reduction on the harmonic component
            # Convert to frequency domain
            print("Applying vocal reduction filter...")
            D = librosa.stft(y_harmonic)
            
            # Get magnitude and phase
            magnitude, phase = librosa.magphase(D)
            
            # Create a spectral mask that reduces frequencies in the speech range
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
            mask = np.ones_like(magnitude)
            
            # Identify bins in the vocal frequency range (200-3500 Hz)
            vocal_bins = np.where((freq_bins >= 200) & (freq_bins <= 3500))[0]
            
            # Reduce these frequencies (but don't eliminate completely)
            for i in vocal_bins:
                if i < mask.shape[0]:
                    mask[i, :] = 0.4  # Reduce by 60%
            
            # Apply the mask
            magnitude_masked = magnitude * mask
            
            # Reconstruct the signal
            D_masked = magnitude_masked * phase
            y_harmonic_filtered = librosa.istft(D_masked)
            
            # Mix with some percussive content for a more balanced sound
            print("Mixing harmonic and percussive components...")
            music = y_harmonic_filtered * 0.8 + y_percussive * 0.5
            
            # Apply a band-pass filter to focus on typical music frequencies
            print("Applying frequency filtering...")
            b, a = butter_bandpass(lowcut=60, highcut=10000, fs=sr, order=4)
            filtered_music = filtfilt(b, a, music)
            
            # Apply dynamic range compression
            print("Applying dynamic range compression...")
            compressed_music = apply_simple_compression(filtered_music, threshold=0.5, ratio=1.5)
            
            # Boost the volume for better audibility
            compressed_music = compressed_music * 2.0
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(compressed_music))
            if max_val > 0:
                compressed_music = compressed_music / max_val * 0.9
            
            sf.write(output_path, compressed_music, sr)
            print(f"Enhanced fallback music extraction saved to {output_path}")
            return output_path
        except Exception as e2:
            print(f"Fallback music extraction also failed: {e2}")
            return None

def apply_spectral_vocal_reduction(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Apply spectral masking to further reduce any vocal content."""
    # If audio is mono, convert to stereo-like format for processing
    if len(audio.shape) == 1:
        audio = np.vstack([audio, audio])
    
    # Get the magnitude spectrogram
    S_left = np.abs(librosa.stft(audio[0]))
    S_right = np.abs(librosa.stft(audio[1]))
    
    # Vocal frequencies are typically between 200Hz and 3000Hz
    # Create a mask that reduces these frequencies
    freq_bins = librosa.fft_frequencies(sr=sr)
    vocal_mask = np.ones(len(freq_bins))
    
    # Identify vocal frequency ranges (approximate)
    vocal_range = (freq_bins >= 200) & (freq_bins <= 3000)
    
    # Reduce the intensity of frequencies in the vocal range
    vocal_mask[vocal_range] = 0.5  # Reduce by 50%
    
    # Apply the mask to the spectrograms
    S_left_masked = S_left * vocal_mask[:, np.newaxis]
    S_right_masked = S_right * vocal_mask[:, np.newaxis]
    
    # Convert back to time domain
    y_left = librosa.griffinlim(S_left_masked, n_iter=32)
    y_right = librosa.griffinlim(S_right_masked, n_iter=32)
    
    return np.vstack([y_left, y_right])

def apply_dynamic_compression(audio: np.ndarray, threshold: float = 0.5, ratio: float = 2.0) -> np.ndarray:
    """Apply dynamic range compression to audio.
    
    Args:
        audio: Audio signal as numpy array
        threshold: Compression threshold (0.0 to 1.0), signals above this level will be compressed
        ratio: Compression ratio (higher values = more compression)
        
    Returns:
        Compressed audio signal
    """
    # If audio is mono, convert to stereo-like format for processing
    if len(audio.shape) == 1:
        audio = np.vstack([audio, audio])
    
    # Process each channel
    compressed_audio = np.zeros_like(audio)
    for i in range(audio.shape[0]):
        # Normalize to -1 to 1 range if not already
        max_val = np.max(np.abs(audio[i]))
        if max_val > 0:
            normalized = audio[i] / max_val
        else:
            normalized = audio[i]
        
        # Apply compression
        compressed = np.zeros_like(normalized)
        for j in range(len(normalized)):
            if abs(normalized[j]) > threshold:
                if normalized[j] > 0:
                    compressed[j] = threshold + (normalized[j] - threshold) / ratio
                else:
                    compressed[j] = -threshold + (normalized[j] + threshold) / ratio
            else:
                compressed[j] = normalized[j]
        
        # Renormalize to original scale
        if max_val > 0:
            compressed_audio[i] = compressed * max_val
        else:
            compressed_audio[i] = compressed
    
    return compressed_audio

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5) -> Tuple:
    """Create a bandpass Butterworth filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """Apply a bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def apply_simple_compression(audio: np.ndarray, threshold: float = 0.3, ratio: float = 2.0) -> np.ndarray:
    """Apply simple dynamic range compression."""
    compressed = np.zeros_like(audio)
    for i in range(len(audio)):
        if abs(audio[i]) > threshold:
            if audio[i] > 0:
                compressed[i] = threshold + (audio[i] - threshold) / ratio
            else:
                compressed[i] = -threshold + (audio[i] + threshold) / ratio
        else:
            compressed[i] = audio[i]
    return compressed

def mix_audio_tracks(speech_path: str, music_path: str, output_path: str, speech_volume: float = 1.0, music_volume: float = 0.5) -> str:
    """
    Mix speech and music audio tracks with volume control and dynamic processing.
    
    This function combines a speech track with a music track, applying volume adjustments,
    dynamic range compression, and equalization to create a balanced mix where speech
    is clearly audible over the background music.
    
    Algorithm:
    1. Load the speech and music audio files
    2. Resample both to the same sample rate if needed
    3. Apply speech clarity equalization to enhance intelligibility
    4. Find silent segments in the speech track
    5. Boost music volume during silent segments for a more natural sound
    6. Apply volume adjustments to both tracks
    7. Mix the tracks together
    8. Apply soft clipping to prevent digital distortion
    9. Save the mixed audio to the output path
    10. Return the path to the mixed audio file
    """
    print(f"Mixing speech and music tracks with enhanced process...")
    print(f"Speech volume: {speech_volume}, Music volume: {music_volume}")
    
    # Load audio files
    speech, speech_sr = librosa.load(speech_path, sr=None)
    music, music_sr = librosa.load(music_path, sr=None)
    
    # Resample music to match speech sample rate if needed
    if music_sr != speech_sr:
        print(f"Resampling music from {music_sr}Hz to {speech_sr}Hz")
        music = librosa.resample(music, orig_sr=music_sr, target_sr=speech_sr)
    
    # Adjust lengths to match
    if len(music) > len(speech):
        print(f"Trimming music to match speech length")
        music = music[:len(speech)]
    elif len(music) < len(speech):
        # Pad music with zeros to match speech length
        print(f"Padding music to match speech length")
        padding = np.zeros(len(speech) - len(music))
        music = np.concatenate([music, padding])
    
    # Apply dynamic range compression to speech to make it more consistent
    print("Compressing speech dynamic range")
    speech = compress_dynamic_range(speech, threshold=0.3, ratio=2.0)
    
    # Find silent segments in speech (for adaptive music volume)
    silent_segments = find_silent_segments(speech, threshold=0.02, min_length=0.5, sr=speech_sr)
    
    # Create a volume envelope for music (higher during silent speech segments)
    music_envelope = np.ones_like(music) * music_volume
    
    # Increase music volume during silent segments
    print(f"Applying dynamic music volume for {len(silent_segments)} silent segments")
    for start, end in silent_segments:
        if start < len(music_envelope) and end <= len(music_envelope):
            # Fade in and out for smooth transitions
            fade_samples = min(int(0.1 * speech_sr), (end - start) // 4)  # 100ms fade or 1/4 of segment
            
            # Only apply if segment is long enough for fades
            if end - start > 2 * fade_samples:
                # Set higher volume in silent segment
                music_envelope[start+fade_samples:end-fade_samples] = music_volume * 1.5
                
                # Create fade in
                fade_in = np.linspace(music_volume, music_volume * 1.5, fade_samples)
                music_envelope[start:start+fade_samples] = fade_in
                
                # Create fade out
                fade_out = np.linspace(music_volume * 1.5, music_volume, fade_samples)
                music_envelope[end-fade_samples:end] = fade_out
    
    # Apply volume adjustments with envelope for music
    speech = speech * speech_volume
    music = music * music_envelope
    
    # Apply a subtle EQ to make speech more clear
    print("Applying speech clarity EQ")
    speech = apply_speech_clarity_eq(speech, speech_sr)
    
    # Mix the tracks
    print("Mixing speech and music tracks")
    
    # Ensure both arrays have exactly the same shape before mixing
    if len(speech) != len(music):
        print(f"Warning: Speech and music arrays have different lengths: speech={len(speech)}, music={len(music)}")
        # Resize to the smaller of the two
        min_length = min(len(speech), len(music))
        speech = speech[:min_length]
        music = music[:min_length]
        print(f"Resized both arrays to length {min_length}")
    
    # Now they have the same shape and can be mixed safely
    mixed = speech + music
    
    # Apply soft clipping to prevent digital distortion but preserve dynamics
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        print(f"Applying soft clipping (max value: {max_val:.2f})")
        mixed = soft_clip(mixed / max_val) * 0.95  # Leave some headroom
    
    # Save the mixed audio
    sf.write(output_path, mixed, speech_sr)
    print(f"Enhanced mixed audio saved to {output_path}")
    
    return output_path

def compress_dynamic_range(audio: np.ndarray, threshold: float = 0.3, ratio: float = 2.0) -> np.ndarray:
    """Apply dynamic range compression to audio.
    
    Args:
        audio: Audio signal as numpy array
        threshold: Compression threshold (0.0 to 1.0), signals above this level will be compressed
        ratio: Compression ratio (higher values = more compression)
        
    Returns:
        Compressed audio signal
    """
    # Handle both mono and stereo efficiently
    is_stereo = len(audio.shape) > 1
    
    # Find samples above threshold
    mask = np.abs(audio) > threshold
    
    # Apply compression to those samples
    audio_compressed = audio.copy()
    
    if is_stereo:
        # Process stereo channels
        for i in range(audio.shape[1]):
            channel_mask = mask[:, i] if is_stereo else mask
            audio_compressed[channel_mask, i] = np.sign(audio[channel_mask, i]) * (
                threshold + (np.abs(audio[channel_mask, i]) - threshold) / ratio
            )
    else:
        # Process mono
        audio_compressed[mask] = np.sign(audio[mask]) * (
            threshold + (np.abs(audio[mask]) - threshold) / ratio
        )
    
    # Normalize to prevent clipping but maintain dynamics
    max_val = np.max(np.abs(audio_compressed))
    if max_val > 0.99:
        audio_compressed = audio_compressed / max_val * 0.99
    
    return audio_compressed

def find_silent_segments(audio: np.ndarray, threshold: float = 0.02, min_length: float = 0.5, sr: int = 22050) -> list:
    """Find segments where speech is silent or very quiet."""
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio
    
    # Calculate the envelope (amplitude)
    envelope = np.abs(audio_mono)
    
    # Find segments below threshold
    is_silent = envelope < threshold
    
    # Find contiguous silent segments
    silent_segments = []
    in_silent = False
    start_idx = 0
    min_samples = int(min_length * sr)
    
    for i in range(len(is_silent)):
        if is_silent[i] and not in_silent:
            # Start of a silent segment
            in_silent = True
            start_idx = i
        elif not is_silent[i] and in_silent:
            # End of a silent segment
            in_silent = False
            if i - start_idx >= min_samples:
                silent_segments.append((start_idx, i))
    
    # Check if we ended in a silent segment
    if in_silent and len(is_silent) - start_idx >= min_samples:
        silent_segments.append((start_idx, len(is_silent)))
    
    return silent_segments

def apply_speech_clarity_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply a subtle EQ to enhance speech clarity."""
    # Convert to mono if stereo for processing
    is_stereo = len(audio.shape) > 1
    if is_stereo:
        channels = []
        for i in range(audio.shape[1]):
            # Process each channel
            channel = audio[:, i]
            # Get the spectrogram
            D = librosa.stft(channel)
            
            # Get magnitude and phase
            magnitude, phase = librosa.magphase(D)
            
            # Boost frequencies around 2-4kHz (speech clarity)
            freq_bins = librosa.fft_frequencies(sr=sr)
            clarity_boost = np.ones(len(freq_bins))
            clarity_range = (freq_bins >= 2000) & (freq_bins <= 4000)
            clarity_boost[clarity_range] = 1.3  # 30% boost
            
            # Apply the EQ
            magnitude = magnitude * clarity_boost[:, np.newaxis]
            
            # Reconstruct the signal
            D_modified = magnitude * phase
            channel_eq = librosa.istft(D_modified)
            
            channels.append(channel_eq)
        
        # Combine channels
        audio_eq = np.column_stack(channels)
    else:
        # Get the spectrogram
        D = librosa.stft(audio)
        
        # Get magnitude and phase
        magnitude, phase = librosa.magphase(D)
        
        # Boost frequencies around 2-4kHz (speech clarity)
        freq_bins = librosa.fft_frequencies(sr=sr)
        clarity_boost = np.ones(len(freq_bins))
        clarity_range = (freq_bins >= 2000) & (freq_bins <= 4000)
        clarity_boost[clarity_range] = 1.3  # 30% boost
        
        # Apply the EQ
        magnitude = magnitude * clarity_boost[:, np.newaxis]
        
        # Reconstruct the signal
        D_modified = magnitude * phase
        audio_eq = librosa.istft(D_modified)
    
    return audio_eq

def soft_clip(x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Apply soft clipping to prevent harsh digital clipping while preserving dynamics.
    
    Args:
        x: Audio signal as numpy array
        threshold: Threshold above which soft clipping is applied (0.0 to 1.0)
        
    Returns:
        Soft-clipped audio signal
    """
    # Find the maximum absolute value
    max_val = np.max(np.abs(x))
    
    # If the signal is already within bounds, return it unchanged
    if max_val <= threshold:
        return x
    
    # Apply a more sophisticated soft clipping that preserves more dynamics
    # Handle both mono and stereo
    if len(x.shape) > 1:  # stereo
        clipped = np.zeros_like(x)
        for i in range(x.shape[1]):
            # Only apply clipping to samples that exceed the threshold
            mask = np.abs(x[:, i]) > threshold
            clipped[:, i] = x[:, i].copy()  # Start with original signal
            
            # Apply arctan-based soft clipping to samples above threshold
            # This provides a smoother transition than tanh
            clipped[mask, i] = np.sign(x[mask, i]) * (
                threshold + (np.arctan((np.abs(x[mask, i]) - threshold) * (np.pi/2)) / np.pi)
            )
        return clipped
    else:  # mono
        clipped = x.copy()  # Start with original signal
        mask = np.abs(x) > threshold
        
        # Apply arctan-based soft clipping to samples above threshold
        clipped[mask] = np.sign(x[mask]) * (
            threshold + (np.arctan((np.abs(x[mask]) - threshold) * (np.pi/2)) / np.pi)
        )
        return clipped
