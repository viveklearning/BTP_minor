# pip install librosa soundfile numpy sklearn pyaudio
import librosa
import soundfile as sf
import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pyaudio
import wave

# Statements (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door") are used in training dataset.

# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# # Initialize the Multi Layer Perceptron Classifier
# model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
#                       hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# # Recording user audio
# def recordAudio():
#     chunk = 1024  # Record in chunks of 1024 samples
#     sample_format = pyaudio.paInt16  # 16 bits per sample
#     channels = 1
#     fs = 48000  # Record at 48000 samples per second
#     seconds = 5
#     filename = "Predict-Record-Audio.wav"

#     p = pyaudio.PyAudio()  # Create an interface to PortAudio

#     print('Recording')

#     stream = p.open(format=sample_format,
#                     channels=channels,
#                     rate=fs,
#                     frames_per_buffer=chunk,
#                     input=True)

#     frames = []  # Initialize array to store frames

#     # Store data in chunks for 5 seconds
#     for i in range(0, int(fs / chunk * seconds)):
#         data = stream.read(chunk)
#         frames.append(data)

#     # Stop and close the stream
#     stream.stop_stream()
#     stream.close()
#     # Terminate the PortAudio interface
#     p.terminate()

#     print('Finished recording')

#     # Save the recorded data as a WAV file
#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(channels)
#     wf.setsampwidth(p.get_sample_size(sample_format))
#     wf.setframerate(fs)
#     wf.writeframes(b''.join(frames))
#     wf.close()

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if chroma:
            stft = np.abs(librosa.stft(X))
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# # Load the data and extract features for each sound file
# def load_data(test_size=0.2):
#     x, y = [], []
#     base_directory = r"Dataset/speech-emotion-recognition-ravdess-data"

#     # List to store all .wav files
#     wav_files = []

#     # Traverse the directory to find .wav files
#     for root, dirs, files in os.walk(base_directory):
#         for file in files:
#             if file.endswith('.wav'):
#                 wav_files.append(os.path.join(root, file))

#     # Print the number of files found
#     print(f"Found {len(wav_files)} .wav files.")

#     # Process each file
#     for wav_file in wav_files:
#         file_name = os.path.basename(wav_file)
#         emotion_code = file_name.split("-")[2]
#         emotion = emotions.get(emotion_code, None)

#         if emotion is None:
#             print(f"Skipping file {file_name}: Unknown emotion.")
#             continue

#         if emotion not in observed_emotions:
#             continue

#         # Extract features
#         try:
#             feature = extract_feature(wav_file, mfcc=True, chroma=True, mel=True)
#             x.append(feature)
#             y.append(emotion)
#         except Exception as e:
#             print(f"Error processing {wav_file}: {e}")
#             continue

#     # Check if any samples were loaded
#     if len(x) == 0 or len(y) == 0:
#         raise ValueError("No valid audio files found or no features extracted.")

#     # Print summary of extracted data
#     print(f"Extracted features for {len(x)} samples.")
#     return train_test_split(np.array(x), np.array(y), test_size=test_size, random_state=9)

# # Train model if started program
# def trainModel():

#     # Split the dataset
#     x_train, x_test, y_train, y_test = load_data(test_size=0.25)

#     # Get the shape of the training and testing datasets
#     print((x_train.shape[0], x_test.shape[0]))

#     # Get the number of features extracted
#     print(f'Features extracted: {x_train.shape[1]}')

#     # Train the model
#     model.fit(x_train, y_train)

#     # Predict for the test set
#     y_pred = model.predict(x_test)

#     # Calculate the accuracy of our model
#     accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

#     # Print the accuracy
#     print("Accuracy: {:.2f}%".format(accuracy * 100))

# # Record your audio and predict emotion
# def record_predictAudio():
#     x_predictAudio = []
#     # recordAudio()  # Record audio to predict
#     file = "Predict-Record-Audio.wav"  # Recorded audio filepath
#     featurePredictAudio = extract_feature(file, mfcc=True, chroma=True, mel=True)  # Extract features of recorded audio
#     x_predictAudio.append(featurePredictAudio)
#     with open('trained_model_ser_real_time.pkl', 'rb') as f:
#         model = pickle.load(f)
#     if len(x_predictAudio) > 0:
#         y_predictAudio = model.predict(np.array(x_predictAudio))
#         print("Emotion Predicted: {}".format(y_predictAudio))
#     else:
#         print("No features extracted from the audio file.")

# Predict on pre-recorded audio
def predictAudio():
    # file = input("Please enter path to your file.\n")
    file="record_happy_sad.wav"
    x_predictAudio = []
    featurePredictAudio = extract_feature(file, mfcc=True, chroma=True, mel=True)  # Extract features of recorded audio
    x_predictAudio.append(featurePredictAudio)
    with open('trained_model_ser_real_time.pkl', 'rb') as f:
        model = pickle.load(f)
    if len(x_predictAudio) > 0:
        y_predictAudio = model.predict(np.array(x_predictAudio))
        print("Emotion Predicted: {}".format(y_predictAudio))
    else:
        print("No features extracted from the audio file.")

# Main menu
while True:
    choice = int(input("Enter 1 to create and train model. \nEnter 2 to record and predict audio. \nEnter 3 to predict on pre-recorded audio. \nEnter 4 to quit. \n"))
    if choice == 1:
        trainModel()
    elif choice == 2:
        record_predictAudio()
    elif choice == 3:
        predictAudio()
    elif choice == 4:
        print("Exiting the program.")
        break
    else:
        print("Invalid choice. Please try again.")
