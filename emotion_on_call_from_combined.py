import cv2
from deepface import DeepFace
from collections import Counter

# Predefined scores for each emotion
emotion_scores = {
    'happy': 10,
    'sad': 5,
    'angry': 4,
    'surprise': 8,
    'fear': 6,
    'disgust': 3,
    'neutral': 7
}

def detect_emotions_in_video(video_path):
    """
    Function to detect emotions in a video file using DeepFace and calculate an average score.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        tuple: A tuple containing a dictionary with emotion counts and the average score.
    """
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the video file
    cap = cv2.VideoCapture(video_path)
    emotions = []
    scores = []

    frame_count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Break the loop if the video ends
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emotions.append(emotion)

                # Assign score to the detected emotion and append to scores list
                score = emotion_scores.get(emotion, 0)  # Default to 0 if emotion is not found in the dictionary
                scores.append(score)

            except Exception as e:
                print(f"Error analyzing face: {e}")
                continue

        frame_count += 1

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Calculate the average score
    average_score = sum(scores) / len(scores) if scores else 0

    # Return emotion count summary and average score
    emotion_summary = Counter(emotions)
    return emotion_summary, average_score
