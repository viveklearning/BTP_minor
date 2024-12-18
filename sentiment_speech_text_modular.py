import numpy as np
import whisper
from transformers import pipeline
import wave
from pydub import AudioSegment

# Constants
RATE = 16000  # Whisper model expects 16000 Hz sampling rate

# Load Whisper model
speech_model = whisper.load_model("base")  # Or "small", "large", etc.

# Sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis")

def transcribe_audio(audio_data):
    # Whisper expects audio in a NumPy array, with normalized values between [-1.0, 1.0]
    audio_data = np.frombuffer(audio_data, dtype=np.int16)  # Ensure it's in 16-bit PCM format
    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize audio data

    # Use Whisper to transcribe the audio (audio_data is now a valid numpy array)
    result = speech_model.transcribe(audio_data)
    return result['text']

def analyze_sentiment(text):
    return sentiment_model(text)

def load_and_resample_audio(file_path):
    try:
        audio = AudioSegment.from_wav(file_path)
        if audio.frame_rate != RATE:
            audio = audio.set_frame_rate(RATE)
        audio = audio.set_channels(1).set_sample_width(2)
        return audio.raw_data
    except Exception as e:
        print(f"Error loading or resampling audio: {e}")
        return None

def process_audio(file_path):
    audio_data = load_and_resample_audio(file_path)
    if audio_data is None:
        print("Failed to load and process audio file.")
        return

    transcribed_text = transcribe_audio(audio_data)
    print(f"Transcribed Text: {transcribed_text}")

    if transcribed_text.strip():
        sentiment = analyze_sentiment(transcribed_text)
        print(f"Sentiment: {sentiment[0]['label']} (Confidence: {sentiment[0]['score']:.2f})")
