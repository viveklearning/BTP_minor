# import numpy as np
# import pyaudio
# import whisper
# from transformers import pipeline
# import io
# import wave

# # Constants
# CHUNK = 1024  # Size of audio chunk to read
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000  # Whisper model expects 16000 Hz sampling rate
# DEVICE_INDEX = None  # Set to None or the appropriate device index if needed

# # Initialize pyaudio stream
# audio = pyaudio.PyAudio()
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=DEVICE_INDEX)

# print("Listening...")

# # Load Whisper model
# speech_model = whisper.load_model("base")  # Or "small", "large", etc.

# # Sentiment analysis pipeline
# sentiment_model = pipeline("sentiment-analysis")

# def transcribe_audio(audio_data):
#     # Convert the audio data to a NumPy array (which is what Whisper expects)
#     audio_data = np.frombuffer(audio_data, dtype=np.int16)  # Ensure it's in 16-bit PCM format
    
#     # Normalize to the range of [-1.0, 1.0] as expected by Whisper
#     audio_data = audio_data.astype(np.float32) / 32768.0

#     # Use Whisper to transcribe the audio (audio_data is now a valid numpy array)
#     result = speech_model.transcribe(audio_data)
#     return result['text']

# def analyze_sentiment(text):
#     return sentiment_model(text)

# try:
#     while True:
#         # Read audio chunk from the microphone
#         data = stream.read(CHUNK)
#         print(data)
#         # Convert the audio chunk to text using Whisper
#         transcribed_text = transcribe_audio(data)
#         print(f"Transcribed Text: {transcribed_text}")

#         if transcribed_text.strip():  # Proceed if there is meaningful transcribed text
#             # Perform sentiment analysis on the transcribed text
#             sentiment = analyze_sentiment(transcribed_text)
#             print(f"Sentiment: {sentiment[0]['label']} (Confidence: {sentiment[0]['score']:.2f})")

# except KeyboardInterrupt:
#     print("Stopping...")
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()


import numpy as np
import whisper
from transformers import pipeline
import wave
from pydub import AudioSegment
import torch

# Constants
AUDIO_FILE_PATH = "record_sad_happy.wav"  # Replace with your audio file path
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

# Load and possibly resample the prerecorded audio file
def load_and_resample_audio(file_path):
    try:
        # Load audio file using pydub
        audio = AudioSegment.from_wav(file_path)

        # Resample the audio to the desired sample rate (16kHz) if it's not already at that rate
        if audio.frame_rate != RATE:
            audio = audio.set_frame_rate(RATE)

        # Convert the audio to a format that Whisper can process (16-bit PCM)
        audio = audio.set_channels(1).set_sample_width(2)  # 2 bytes per sample = 16-bit
        audio_data = audio.raw_data  # Get raw audio data in bytes
        return audio_data
    except Exception as e:
        print(f"Error loading or resampling audio: {e}")
        return None

# Load audio file and ensure it's in the correct format
def load_audio_file(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            # Ensure the audio is in 16kHz mono format (the expected input format for Whisper)
            assert wf.getframerate() == RATE, f"Expected sample rate {RATE}, but got {wf.getframerate()}"
            assert wf.getnchannels() == 1, f"Expected mono audio, but got {wf.getnchannels()} channels"

            # Read the audio data
            audio_data = wf.readframes(wf.getnframes())
            return audio_data
    except Exception as e:
        print(f"Error processing the audio file: {e}")
        return None

# Main processing loop for prerecorded audio
def process_audio(file_path):
    audio_data = load_and_resample_audio(file_path)
    if audio_data is None:
        print("Failed to load and process audio file.")
        return

    # Transcribe audio to text using Whisper
    transcribed_text = transcribe_audio(audio_data)
    print(f"Transcribed Text: {transcribed_text}")

    if transcribed_text.strip():  # Proceed if there is meaningful transcribed text
        # Perform sentiment analysis on the transcribed text
        sentiment = analyze_sentiment(transcribed_text)
        print(f"Sentiment: {sentiment[0]['label']} (Confidence: {sentiment[0]['score']:.2f})")

# Run the processing
if __name__ == "__main__":
    process_audio(AUDIO_FILE_PATH)
