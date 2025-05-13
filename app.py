"""
Streamlit GUI for Video Dubbing Tool
Supports English, Arabic, and Japanese video dubbing using Whisper and Coqui TTS.
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
    'English': 'en',
    'Arabic': 'ar',
    'Japanese': 'ja',
}

# Fixed model settings
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
TARGET_LANGUAGE = "en"  # Always dub to English

def main():
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
        
        # Speaker selection (if available)
        speakers = list_speakers_for_model(TTS_MODEL)
        speaker = st.selectbox(
            "Speaker Voice",
            speakers if speakers else ["Default"],
            index=0
        ) if speakers else None
        
        # Audio processing options
        preserve_music = st.checkbox("Preserve background music", value=True)
        apply_noise_reduction = st.checkbox("Apply noise reduction", value=True)
        
        # Volume controls
        st.subheader("Volume Controls")
        speech_volume = st.slider("Speech Volume", 0.1, 2.0, 1.0, 0.1)
        music_volume = st.slider("Music Volume", 0.0, 1.0, 0.3, 0.05)
        
        # Advanced info
        st.markdown("---")
        st.markdown("### Technical Info")
        st.markdown("- ASR: Whisper Medium")
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
                                progress_bar.progress(step / total_steps)
                                status_messages = [
                                    "Extracting audio...",
                                    "Transcribing with Whisper...",
                                    "Synthesizing speech...",
                                    "Mixing audio...",
                                    "Merging with video...",
                                    "Finalizing..."
                                ]
                                status_text.text(status_messages[step-1])
                                time.sleep(0.5)  # For visual feedback
                            
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
                            
                            # Process the video (this actually does all the steps)
                            process_video(
                                video_path=input_path,
                                output_path=output_path,
                                source_language=SUPPORTED_LANGUAGES[source_lang],
                                preserve_music=preserve_music,
                                apply_noise_red=apply_noise_reduction,
                                speech_volume=speech_volume,
                                music_volume=music_volume,
                                tts_model=TTS_MODEL,
                                speaker=speaker if speaker != "Default" else None,
                                target_language=TARGET_LANGUAGE
                            )
                            
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
