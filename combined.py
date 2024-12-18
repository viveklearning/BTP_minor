from moviepy.editor import VideoFileClip
from emotion_on_call_from_combined import detect_emotions_in_video
from Ser_real_time_saved_video import predictAudio
from sentiment_speech_text_modular import process_audio
import os

def convert_video_to_wav(video_path, output_wav_path):
    try:
        # Load the video file
        video_clip = VideoFileClip(video_path)
        
        # Extract audio and write to WAV file
        video_clip.audio.write_audiofile(output_wav_path, codec='pcm_s16le')
        print(f"Audio extracted and saved to {output_wav_path}")
    except Exception as e:
        print(f"Error converting video to WAV: {e}")
    finally:
        # Ensure resources are released
        video_clip.close()

if __name__ == "__main__":
    video_path = 'video2.mp4'  # Replace with your actual video file path
    wav_path = 'temp_audio.wav'  # Path for the extracted WAV file

    # Step 1: Convert video to WAV
    video_path = 'video2.mp4'  # Replace with your actual video file path
    # Provide the path to your audio file
    audio_file = "temp_audio.wav"
    print("The emotions detected in the video are : \n")
    print(detect_emotions_in_video(video_path))

    # print("Calling the predictAudio function from another file:")
    print("The emotions detected in the audio information is : \n")

    predictAudio('Predict-Record-Audio.wav')
    print("The emotions detected in the audio speech text is : \n")
    process_audio(audio_file)

    # # Optional: Cleanup the extracted audio file if not needed
    # if os.path.exists(wav_path):
    #     os.remove(wav_path)
    #     print(f"Temporary file {wav_path} removed.")

# # File: main.py

# from emotion_on_call_from_combined import detect_emotions_in_video
# from Ser_real_time_saved_video import predictAudio
# # Call the function with the video file path

# from sentiment_speech_text_modular import process_audio



# # Call the process_audio function



# if __name__ == "__main__":
#     video_path = 'video2.mp4'  # Replace with your actual video file path
#     # Provide the path to your audio file
#     audio_file = "record_sad_happy.wav"
#     detect_emotions_in_video(video_path)
#     print("Calling the predictAudio function from another file:")
#     predictAudio('Predict-Record-Audio.wav')
#     process_audio(audio_file)



