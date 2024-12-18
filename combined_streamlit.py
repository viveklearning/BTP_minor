import streamlit as st
from moviepy.editor import VideoFileClip
from emotion_on_call_from_combined import detect_emotions_in_video
from Ser_real_time_saved_video import predictAudio
from sentiment_speech_text_modular import process_audio
import os

def convert_video_to_wav(video_path, output_wav_path):
    """Extracts audio from a video file and saves it as a WAV file."""
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(output_wav_path, codec='pcm_s16le')
        return True, f"Audio extracted and saved to {output_wav_path}"
    except Exception as e:
        return False, f"Error converting video to WAV: {e}"
    finally:
        video_clip.close()

def main():
    st.title("Multimodal Emotion Analysis App")
    st.write("Upload a video to analyze emotions from both video and audio.")

    # Upload video file
    uploaded_file = st.file_uploader("Upload a video file (e.g., .mp4)", type=["mp4", "mov", "avi"])

    if uploaded_file:
        video_path = f"uploaded_{uploaded_file.name}"
        output_wav_path = "extracted_audio.wav"

        # Save the uploaded video file locally
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Video '{uploaded_file.name}' uploaded successfully.")

        # Step 1: Convert video to WAV
        st.write("Extracting audio from the video...")
        success, message = convert_video_to_wav(video_path, output_wav_path)
        if success:
            st.success(message)
        else:
            st.error(message)
            return

        # Step 2: Run video emotion analysis
        st.write("Running emotion analysis on the video...")
        try:
            detect_emotions_in_video(video_path)
            st.success("Video emotion analysis completed successfully!")
        except Exception as e:
            st.error(f"Error in video emotion analysis: {e}")

        # Step 3: Run audio emotion analysis
        st.write("Running emotion analysis on the audio...")
        try:
            predictAudio(output_wav_path)
            st.success("Audio emotion analysis completed successfully!")
        except Exception as e:
            st.error(f"Error in audio emotion analysis: {e}")

        # Step 4: Sentiment analysis of audio transcription
        st.write("Running sentiment analysis on the audio transcription...")
        try:
            process_audio(output_wav_path)
            st.success("Sentiment analysis completed successfully!")
        except Exception as e:
            st.error(f"Error in sentiment analysis: {e}")

        # Cleanup temporary files
        st.write("Cleaning up temporary files...")
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(output_wav_path):
                os.remove(output_wav_path)
            st.success("Temporary files cleaned up!")
        except Exception as e:
            st.warning(f"Error cleaning up files: {e}")

if __name__ == "__main__":
    main()
