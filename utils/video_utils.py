"""
Video utility functions for video dubbing tool.
Handles merging dubbed audio and music with video.
"""
import os
from typing import Optional
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from utils.audio_utils import mix_audio_tracks

def merge_audio_with_video(
    video_path: str, 
    speech_path: str, 
    output_path: str, 
    music_path: Optional[str] = None, 
    speech_volume: float = 1.0, 
    music_volume: float = 0.3
) -> str:
    """
    Merge the dubbed speech and optional music with the original video.
    
    Args:
        video_path: Path to the original video file
        speech_path: Path to the dubbed speech audio file
        output_path: Path where the output video will be saved
        music_path: Optional path to the extracted music file
        speech_volume: Volume multiplier for speech (default: 1.0)
        music_volume: Volume multiplier for music (default: 0.3)
        
    Returns:
        Path to the output video file
    """
    print(f"Merging audio with video: {video_path}")
    video = VideoFileClip(video_path)
    
    if music_path and os.path.exists(music_path):
        # If we have a music track, mix it with the speech
        print("Using extracted music track")
        mixed_path = os.path.join(os.path.dirname(speech_path), "mixed_audio.wav")
        mix_audio_tracks(speech_path, music_path, mixed_path, speech_volume, music_volume)
        audio = AudioFileClip(mixed_path)
    else:
        # Otherwise just use the speech track
        print("No music track available, using speech only")
        audio = AudioFileClip(speech_path)
    
    # Set the audio to the video
    # Note: set_audio returns a new clip, it doesn't modify the original
    video_with_audio = video.set_audio(audio)
    
    # Write the output video file
    print(f"Writing output video to: {output_path}")
    video_with_audio.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    
    # Close all clips to free resources
    video.close()
    audio.close()
    video_with_audio.close()
    
    print(f"Video with dubbed audio saved to: {output_path}")
    return output_path
