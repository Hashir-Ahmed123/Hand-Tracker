import cv2
import mediapipe as mp
import pyaudio
import numpy as np

# Initialize Mediapipe and PyAudio
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# PyAudio parameters
p = pyaudio.PyAudio()
volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 1.0   # in seconds, may be float
f = 440.0        # sine frequency, Hz, may be float

# Generate samples
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

# Stream callback function
def callback(in_data, frame_count, time_info, status):
    return (volume * samples).tobytes(), pyaudio.paContinue

# Open stream
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True,
                stream_callback=callback)

# Start the stream
stream.start_stream()

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # Convert the image color back so it can be displayed
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check the state of the gesture
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            if (thumb_tip.x < index_finger_tip.x and thumb_tip.x < middle_finger_tip.x):
                # Gesture: All fingers together (lower volume)
                volume = 0.1
            else:
                # Gesture: Fingers apart (raise volume)
                volume = 1.0
    
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()
