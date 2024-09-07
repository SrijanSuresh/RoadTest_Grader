import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load pre-trained MobileNet model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load and preprocess the image (for live camera input, use OpenCV)
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Resize to model input size
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Get a frame from the camera
cap = cv2.VideoCapture(0)  # Open webcam
ret, frame = cap.read()

if ret:
    img = preprocess_image(frame)
    predictions = model.predict(img)
    
    # Get the predicted class label (Seatbelt On/Off)
    print('Prediction:', tf.keras.applications.mobilenet_v2.decode_predictions(predictions))

cap.release()
cv2.destroyAllWindows()
