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
    music_volume: float = 0.3,
    fps: Optional[float] = None
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
        fps: Optional frame rate for output video
        
    Returns:
        Path to the output video file
    """
    print(f"Merging audio with video: {video_path}")
    print(f"Speech volume: {speech_volume}, Music volume: {music_volume}")
    
    # Get original video properties
    video = VideoFileClip(video_path)
    original_duration = video.duration
    original_fps = video.fps
    
    # Use provided FPS or original
    if fps is None:
        fps = original_fps
    
    # Load the speech audio
    speech_audio = AudioFileClip(speech_path)
    
    # Create a mixed audio file if music is available
    if music_path and os.path.exists(music_path):
        print(f"Using extracted music track: {music_path}")
        
        # Option 1: Use the mix_audio_tracks function for better audio mixing
        temp_mixed_path = os.path.join(os.path.dirname(output_path), "temp_mixed_audio.wav")
        mix_audio_tracks(
            speech_path=speech_path,
            music_path=music_path,
            output_path=temp_mixed_path,
            speech_volume=speech_volume,
            music_volume=music_volume
        )
        
        # Load the mixed audio
        audio = AudioFileClip(temp_mixed_path)
        print(f"Created enhanced mixed audio with speech and music")
    else:
        # Otherwise just use the speech track
        print("No music track available, using speech only")
        audio = speech_audio.volumex(speech_volume)
    
    # Ensure audio duration matches video duration
    if abs(audio.duration - original_duration) > 0.1:  # If difference is more than 0.1 seconds
        print(f"Adjusting audio duration from {audio.duration:.2f}s to match video duration {original_duration:.2f}s")
        
        # If audio is shorter than video, loop it or add silence to match
        if audio.duration < original_duration:
            # Calculate how many times to loop the audio
            loop_count = int(original_duration / audio.duration) + 1
            print(f"Audio is shorter than video, extending by looping {loop_count} times")
            
            # Create a list of audio clips to concatenate
            audio_clips = [audio] * loop_count
            extended_audio = CompositeAudioClip(audio_clips)
            
            # Trim to exact video duration
            audio = extended_audio.subclip(0, original_duration)
        else:
            # If audio is longer, just trim it
            audio = audio.subclip(0, original_duration)
    
    # Set the audio to the video
    # Note: set_audio returns a new clip, it doesn't modify the original
    video_with_audio = video.set_audio(audio)
    
    # Write the output video file with higher quality settings
    print(f"Writing output video to: {output_path}")
    video_with_audio.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        audio_bitrate="192k",  # Higher audio bitrate for better quality
        bitrate="8000k",      # Higher video bitrate
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        fps=fps,  # Use the original or specified FPS
        verbose=False,
        logger=None
    )
    
    # Clean up temporary files
    if music_path and os.path.exists(music_path) and os.path.exists(temp_mixed_path):
        try:
            os.remove(temp_mixed_path)
            print(f"Removed temporary mixed audio file")
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")
    
    # Close all clips to free resources
    video.close()
    audio.close()
    video_with_audio.close()
    
    print(f"Video with dubbed audio saved to: {output_path}")
    return output_path
