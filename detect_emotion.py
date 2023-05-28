import cv2
import numpy as np
from deepface import DeepFace
import matplotlib as plt


def detect_emotions(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face = cv2.resize(face_roi, (152, 152))  # Resize to match the input shape of the model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        face = np.expand_dims(face, axis=0)
        face = face.astype('float32') / 255.0

        # emotion_predictions = model.predict(face)[0, :]
        # emotion_index = np.argmax(emotion_predictions)
        # # emotion_label = emotion_labels[emotion_index]
        # print(emotion_index)
        # print("dekh")
        # print(emotion_predictions)


        # Perform race and gender detection
        detection = DeepFace.analyze(face_roi, actions=['race', 'gender', 'emotion'], enforce_detection=False)

        if len(detection) > 0:
            detection = detection[0]  # Access the first item in the list

            race_label = detection['dominant_race']
            gender_label = detection['dominant_gender']
            emotion = detection['dominant_emotion']
            print(detection)

            # Format the labels to display on the screen
            labels = f"A {gender_label} showing {emotion} emotion"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, labels, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return frame
