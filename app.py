"""
Streamlit GUI for Video Dubbing Tool

This application provides a user-friendly interface for dubbing videos into English
using Whisper for speech recognition and Coqui TTS for speech synthesis. It supports
multiple source languages and includes voice cloning capability to maintain the original
speaker's voice characteristics in the dubbed version.

Features:
- Automatic speech recognition and translation using Whisper
- Text-to-speech synthesis using Coqui TTS models
- Voice cloning option to preserve the original speaker's voice
- Background music preservation
- Audio processing options (noise reduction, volume control)
- Temporary file management for debugging

Supported languages: Arabic, Japanese, and auto-detection
"""
import streamlit as st
import tempfile
import os
import time
from pathlib import Path
from utils.dubbing_utils import process_video
from utils.tts_utils import list_speakers_for_model

# Define supported languages
SUPPORTED_LANGUAGES = {
    'Auto Detect': None,
    'Arabic': 'ar',
    'Japanese': 'ja',
}

# Fixed model settings
DEFAULT_TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
VOICE_CLONING_MODEL = "tts_models/multilingual/multi-dataset/your_tts"  # Always use YourTTS for voice cloning
TARGET_LANGUAGE = "en"  # Always dub to English

def main():
    """
    Main application function that sets up the Streamlit UI and handles the video dubbing workflow.
    
    This function creates the user interface with Streamlit, processes user inputs,
    and orchestrates the video dubbing process. It includes the following steps:
    1. Set up the page configuration and layout
    2. Create sidebar with options for language, voice settings, and audio processing
    3. Handle file uploads and display the original video
    4. Process the video when the user clicks the dubbing button
    5. Display progress updates during processing
    6. Show the dubbed video and provide download option when complete
    
    Returns:
        None
    """
    st.set_page_config(
        page_title="Video Dubbing Tool",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Video Dubbing Tool")
    st.subheader("Dub your videos into English with AI")
    
    # Sidebar for options
    with st.sidebar:
        st.header("Options")
        
        # Source language selection
        source_lang = st.selectbox(
            "Source Language",
            list(SUPPORTED_LANGUAGES.keys()),
            index=0
        )
        
        # Voice cloning options
        st.subheader("Voice Settings")
        clone_voice = st.checkbox("Clone original voice", value=False, 
                                help="Use AI to clone the original voice from the video")
        
        # TTS model selection based on cloning choice
        if clone_voice:
            # Use YourTTS for voice cloning
            tts_model = VOICE_CLONING_MODEL
        else:
            # Standard TTS model
            tts_model = DEFAULT_TTS_MODEL
        
        # Audio processing options
        st.subheader("Audio Processing")
        preserve_music = st.checkbox("Preserve background music", value=True)
        apply_noise_reduction = st.checkbox("Apply noise reduction", value=True)
        keep_temp_files = st.checkbox("Save temporary files", value=False, 
                                    help="Save intermediate files (extracted audio, music, etc.) for review")
        
        # Volume controls
        st.subheader("Volume Controls")
        speech_volume = st.slider("Speech Volume", 0.0, 2.0, 1.5, 0.1)
        music_volume = st.slider("Music Volume", 0.0, 2.0, 1.0, 0.1)
        
        # Advanced info
        st.markdown("---")
        st.markdown("### Technical Info")
        st.markdown("- ASR: Whisper Medium")
        
        # Show appropriate TTS model info
        if clone_voice:
            st.markdown("- TTS: YourTTS (Voice Cloning)")
        else:
            st.markdown("- TTS: Tacotron2-DDC")
            
        st.markdown("- Target Language: English")
        st.markdown("- GPU Acceleration: Enabled")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your video file",
            type=["mp4", "mov", "avi", "mkv"],
            help="Upload a video file to dub into English"
        )
        
        if uploaded_file:
            # Display video info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB"
            }
            st.json(file_details)
            
            # Show original video
            st.subheader("Original Video")
            st.video(uploaded_file)
    
    with col2:
        if uploaded_file:
            # Process button
            if st.button("Start Dubbing", type="primary"):
                with st.spinner("Processing video..."):
                    # Create a temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save uploaded file
                        input_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(input_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Create output path
                        output_filename = f"dubbed_{Path(uploaded_file.name).stem}.mp4"
                        output_path = os.path.join(temp_dir, output_filename)
                        
                        # Process the video
                        try:
                            # Progress indicators
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Update status
                            def update_status(step, total_steps=6):
                                """
                                Update the progress bar and status message during video processing.
                                
                                Args:
                                    step (int): Current processing step (1-6)
                                    total_steps (int): Total number of steps in the process (default: 6)
                                    
                                Returns:
                                    None
                                """
                                progress = step / total_steps
                                progress_bar.progress(progress)
                                
                                status_messages = {
                                    1: "Extracting audio...",
                                    2: "Applying noise reduction...",
                                    3: "Transcribing and translating...",
                                    4: "Synthesizing speech...",
                                    5: "Merging audio with video...",
                                    6: "Finalizing..."
                                }
                                
                                status_text.text(status_messages.get(step, ""))
                            
                            # Step 1: Extract audio
                            update_status(1)
                            
                            # Step 2: Transcribe
                            update_status(2)
                            
                            # Step 3: Synthesize speech
                            update_status(3)
                            
                            # Step 4: Mix audio
                            update_status(4)
                            
                            # Step 5: Merge with video
                            update_status(5)
                            
                            # Create output directory for saving files
                            temp_dir_path = None
                            if keep_temp_files:
                                # Create an output folder in the app directory
                                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                                os.makedirs(output_dir, exist_ok=True)
                                
                                # Create a subdirectory with timestamp
                                import datetime
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                temp_dir_path = os.path.join(output_dir, f"dubbing_{timestamp}")
                                status_text.text(f"Saving files to: {temp_dir_path}")
                            
                            try:
                                # Process the video (this actually does all the steps)
                                process_video(
                                    video_path=input_path,
                                    output_path=output_path,
                                    source_language=SUPPORTED_LANGUAGES[source_lang],
                                    preserve_music=preserve_music,
                                    apply_noise_red=apply_noise_reduction,
                                    speech_volume=speech_volume,
                                    music_volume=music_volume,
                                    tts_model=tts_model,
                                    speaker=None,
                                    target_language=TARGET_LANGUAGE,
                                    clone_voice=clone_voice,
                                    voice_sample_duration=10.0,
                                    keep_temp_files=keep_temp_files,
                                    temp_dir_path=temp_dir_path
                                )
                            except Exception as e:
                                # Show error message
                                st.error(f"Error during dubbing: {str(e)}")
                                st.exception(e)
                                # Exit the try block
                                raise e
                            
                            # Show temp files location if saved
                            if keep_temp_files and temp_dir_path:
                                st.info(f"Temporary files saved to: {temp_dir_path}")
                                # List the saved files
                                temp_files = os.listdir(temp_dir_path)
                                if temp_files:
                                    st.expander("Saved temporary files", expanded=False).write("\n".join(temp_files))
                            
                            # Step 6: Finalize
                            update_status(6)
                            
                            # Read the output file
                            with open(output_path, "rb") as f:
                                output_bytes = f.read()
                            
                            # Complete
                            progress_bar.progress(1.0)
                            status_text.text("Dubbing complete!")
                            
                            # Display the result
                            st.subheader("Dubbed Video")
                            st.video(output_bytes)
                            
                            # Download button
                            st.download_button(
                                label="Download Dubbed Video",
                                data=output_bytes,
                                file_name=output_filename,
                                mime="video/mp4"
                            )
                            
                        except Exception as e:
                            st.error(f"Error during dubbing: {str(e)}")
                            st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("Video Dubbing Tool - Powered by Whisper and Coqui TTS")

if __name__ == "__main__":
    main()
