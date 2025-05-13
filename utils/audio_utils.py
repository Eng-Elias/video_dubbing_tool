"""
Audio utility functions for video dubbing tool.
Handles audio extraction, noise reduction, and music extraction.
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
    """Extract audio from video file."""
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_extracted.wav"
    
    print(f"Extracting audio from {video_path}...")
    video = VideoFileClip(video_path)
    # Extract stereo audio if available (for better music separation)
    video.audio.write_audiofile(output_path, codec='pcm_s16le')
    video.close()
    
    return output_path

def apply_noise_reduction(audio_path: str, output_path: Optional[str] = None) -> str:
    """Apply noise reduction to improve transcription accuracy."""
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
    """Extract music from the audio using Demucs with enhanced vocal removal."""
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
        audio, sr = torchaudio.load(audio_path)
        
        # Apply source separation
        print("Applying source separation...")
        # Add a batch dimension
        audio = audio.unsqueeze(0)
        
        # Apply the model
        sources = apply_model(model, audio.to(device), device=device)
        
        # Get the accompaniment (music without vocals)
        # The order is typically: drums, bass, other, vocals
        # We want everything except vocals
        if model.sources[-1] == "vocals":
            # Sum all sources except vocals
            accompaniment = sources[:, :-1].sum(dim=1)
        else:
            # If vocals is not the last source, find its index
            vocal_idx = model.sources.index("vocals")
            # Create a mask for all sources except vocals
            mask = torch.ones(len(model.sources), dtype=torch.bool)
            mask[vocal_idx] = False
            # Sum all sources except vocals
            accompaniment = sources[:, mask].sum(dim=1)
        
        # Apply spectral vocal reduction to further clean up any vocal remnants
        accompaniment = accompaniment.squeeze(0).cpu().numpy()
        
        # Apply additional vocal reduction
        enhanced_music = apply_spectral_vocal_reduction(accompaniment, sr)
        
        # Apply dynamic compression to make the music more consistent
        compressed_music = apply_dynamic_compression(enhanced_music)
        
        # Save the extracted music
        sf.write(output_path, compressed_music.T, sr)
        print(f"Music extracted and saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error using Demucs: {e}")
        print("Falling back to simple audio filtering...")
        
        # Fallback: Use simple filtering to try to preserve music
        y, sr = librosa.load(audio_path, sr=None)
        
        # Apply bandpass filter (music often has more energy in mid-range frequencies)
        lowcut = 100
        highcut = 8000
        y_filt = butter_bandpass_filter(y, lowcut, highcut, sr)
        
        # Apply simple compression to make the music more consistent
        y_compressed = apply_simple_compression(y_filt)
        
        # Save the filtered audio
        sf.write(output_path, y_compressed, sr)
        print(f"Simple filtered audio saved to {output_path}")
        
        return output_path

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

def apply_dynamic_compression(audio: np.ndarray, threshold: float = 0.3, ratio: float = 2.0) -> np.ndarray:
    """Apply simple dynamic range compression."""
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
    """Mix speech and music audio tracks with volume control and dynamic processing."""
    print(f"Mixing speech and music with volumes: speech={speech_volume}, music={music_volume}")
    
    # Load audio files
    speech, speech_sr = sf.read(speech_path)
    
    # Make sure speech is mono
    if len(speech.shape) > 1 and speech.shape[1] > 1:
        speech = np.mean(speech, axis=1)
    
    # Apply compression to speech to make it more consistent
    speech = compress_dynamic_range(speech)
    
    # If music path is provided and exists, mix with speech
    if music_path and os.path.exists(music_path):
        music, music_sr = sf.read(music_path)
        
        # If music is stereo and speech is mono, convert speech to stereo
        if len(music.shape) > 1 and music.shape[1] > 1 and len(speech.shape) == 1:
            speech = np.column_stack((speech, speech))
        
        # If music is mono and speech is stereo, convert music to stereo
        elif len(speech.shape) > 1 and speech.shape[1] > 1 and len(music.shape) == 1:
            music = np.column_stack((music, music))
        
        # Resample music if necessary
        if music_sr != speech_sr:
            music = librosa.resample(music, orig_sr=music_sr, target_sr=speech_sr)
        
        # Make sure both arrays are the same length
        max_len = max(len(speech), len(music))
        if len(speech) < max_len:
            if len(speech.shape) > 1:  # stereo
                speech = np.vstack([speech, np.zeros((max_len - len(speech), speech.shape[1]))])
            else:  # mono
                speech = np.concatenate([speech, np.zeros(max_len - len(speech))])
        if len(music) < max_len:
            if len(music.shape) > 1:  # stereo
                music = np.vstack([music, np.zeros((max_len - len(music), music.shape[1]))])
            else:  # mono
                music = np.concatenate([music, np.zeros(max_len - len(music))])
        
        # Find silent segments in speech
        silent_segments = find_silent_segments(speech, threshold=0.02, min_length=0.5, sr=speech_sr)
        
        # Create a dynamic music volume envelope
        music_envelope = np.ones(len(music)) * music_volume
        
        # Increase music volume during silent segments
        for start_idx, end_idx in silent_segments:
            # Add a small fade in/out to avoid abrupt volume changes
            fade_samples = int(0.1 * speech_sr)  # 100ms fade
            
            # Apply fade in
            start_fade = max(0, start_idx - fade_samples)
            for i in range(start_fade, start_idx):
                ratio = (i - start_fade) / fade_samples
                music_envelope[i] = music_volume + (music_volume * 0.5) * ratio
            
            # Set higher volume during silence
            music_envelope[start_idx:end_idx] = music_volume * 1.5
            
            # Apply fade out
            end_fade = min(len(music_envelope), end_idx + fade_samples)
            for i in range(end_idx, end_fade):
                ratio = 1 - ((i - end_idx) / fade_samples)
                music_envelope[i] = music_volume + (music_volume * 0.5) * ratio
        
        # Apply the envelope to the music
        if len(music.shape) > 1:  # stereo
            for i in range(music.shape[1]):
                music[:, i] = music[:, i] * music_envelope
        else:  # mono
            music = music * music_envelope
        
        # Apply speech clarity enhancement
        speech = apply_speech_clarity_eq(speech, speech_sr)
        
        # Mix the audio
        if len(speech.shape) > 1 and len(music.shape) > 1:  # both stereo
            mixed = speech * speech_volume + music * music_volume
        elif len(speech.shape) > 1:  # speech stereo, music mono
            mixed = np.zeros_like(speech)
            for i in range(speech.shape[1]):
                mixed[:, i] = speech[:, i] * speech_volume + music * music_volume
        elif len(music.shape) > 1:  # speech mono, music stereo
            mixed = np.zeros_like(music)
            for i in range(music.shape[1]):
                mixed[:, i] = speech * speech_volume + music[:, i] * music_volume
        else:  # both mono
            mixed = speech * speech_volume + music * music_volume
    else:
        # No music, just use speech with volume adjustment
        mixed = speech * speech_volume
    
    # Apply soft clipping to prevent digital distortion
    mixed = soft_clip(mixed)
    
    # Save mixed audio
    sf.write(output_path, mixed, speech_sr)
    print(f"Mixed audio saved to {output_path}")
    
    return output_path

def compress_dynamic_range(audio: np.ndarray, threshold: float = 0.3, ratio: float = 2.0) -> np.ndarray:
    """Apply simple dynamic range compression."""
    # Handle both mono and stereo
    if len(audio.shape) > 1:  # stereo
        compressed = np.zeros_like(audio)
        for i in range(audio.shape[1]):
            compressed[:, i] = apply_simple_compression(audio[:, i], threshold, ratio)
        return compressed
    else:  # mono
        return apply_simple_compression(audio, threshold, ratio)

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
    """Apply soft clipping to prevent harsh digital clipping."""
    # Handle both mono and stereo
    if len(x.shape) > 1:  # stereo
        clipped = np.zeros_like(x)
        for i in range(x.shape[1]):
            clipped[:, i] = np.tanh(x[:, i] / threshold) * threshold
        return clipped
    else:  # mono
        return np.tanh(x / threshold) * threshold
