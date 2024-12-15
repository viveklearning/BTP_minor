import cv2
import numpy as np
import librosa
import speech_recognition as sr
from transformers import pipeline
from deepface import DeepFace

# Load models
visual_emotion_model = DeepFace
text_sentiment_model = pipeline("sentiment-analysis")  # Hugging Face transformers
audio_emotion_model = None  # Placeholder for a trained audio emotion recognition model

# Video Processing
# Process video to analyze visual emotions
def process_video(video_path):
    """
    Analyze visual emotions in a video file.
    Args:
        video_path: Path to the video file.
    """
    # Load the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze emotions every nth frame based on the frame rate
        if frame_count % frame_rate == 0:
            emotions = get_visual_emotion(frame)
            print(f"Frame {frame_count}: Detected Emotions: {emotions}")

        frame_count += 1

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import subprocess

# def extract_audio(video_path):
#     audio_path = "temp_audio.wav"
#     command = [
#         "ffmpeg",
#         "-i", video_path,
#         "-ac", "1",
#         audio_path
#     ]
#     subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     return audio_path



# # Helper: Split audio into chunks
# def split_audio(audio_path, chunk_duration=3):
#     audio, sr_rate = librosa.load(audio_path, sr=16000)
#     chunk_samples = chunk_duration * sr_rate
#     return [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]


import cv2

from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Helper: Get visual emotion
def get_visual_emotion(frame):
    """
    Analyze emotions in detected faces within a video frame.
    Args:
        frame: The current video frame (BGR format).
    Returns:
        A list of detected emotions for each face in the frame.
    """
    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    emotions = []
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the face
        face_roi = frame[y:y + h, x:x + w]

        try:
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result['dominant_emotion']
            emotions.append(dominant_emotion)

            # Annotate the frame with the detected emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except Exception as e:
            print(f"Error analyzing emotion: {e}")

    # Display the frame with annotations
    cv2.imshow('Emotion Detection', frame)
    return emotions


# # Helper: Get audio emotion
# def get_audio_emotion(audio_chunk, sr_rate):
#     try:
#         mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sr_rate, n_mfcc=40)
#         # Placeholder: Replace with pre-trained audio emotion recognition model
#         return "Neutral"
#     except Exception as e:
#         return f"Error: {e}"

# # Helper: Get text emotion
# def get_text_emotion(audio_chunk, recognizer):
#     try:
#         with sr.AudioFile(audio_chunk) as source:
#             audio = recognizer.record(source)
#         text = recognizer.recognize_google(audio)
#         sentiment = text_sentiment_model(text)
#         return sentiment[0]['label']
#     except Exception as e:
#         return f"Error: {e}"

# Main Function
if __name__ == "__main__":
    video_path = "video.mp4"
    process_video(video_path)
