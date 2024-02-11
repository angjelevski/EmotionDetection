import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from pickle import load
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = load_model('best_model.h5')
scaler = load(open('scaler.pkl', 'rb'))

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the emotions
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the captured frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Put a rectangle around the face (ROI)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Normalize the image and reshape for model input
        roi = roi_gray.astype("float")  # - 129.38585663) / 65.0578923  # / 255.0
        # roi = roi.reshape(-1, 1)
        # roi = scaler.transform(roi)
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # roi = roi.reshape(-1, 48, 48, 1)

        # Make a prediction
        preds = model.predict(roi)[0]
        print(preds)
        emotion_label = emotion_dict[np.argmax(preds)]
        # emotion_label = "happy"
        # Display the emotion label on the frame
        cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
