from flask import Flask, request, render_template, jsonify, url_for
import uuid
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

# Flask application setup
app = Flask(__name__)

# Load the pre-trained model
model_path = 'deepfake_detection_model.h5'  # Update this path if necessary
print(f"Loading model from {model_path}...")
model = load_model(model_path)
print("Model loaded successfully.")

# Function to extract frames from video
def extract_frames(video_path, num_frames=10):
    print(f"Extracting frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (128, 128))
            frames.append(frame)
    cap.release()
    print(f"Extracted {len(frames)} frames from video: {video_path}")
    return frames

# Route to render home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle video upload and prediction
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    video_id = str(uuid.uuid4())
    video_path = f"/tmp/{video_id}.mp4"
    file.save(video_path)

    # Extract frames from the video
    frames = extract_frames(video_path)
    frame_data = []
    for frame in frames:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')
        frame_data.append(frame_encoded)

    # Predict whether the video is real or fake
    label = predict_video(frames)
    return render_template('result.html', label=label, frames=frame_data)

# Test the model on a new video
def predict_video(frames):
    frames = np.array(frames) / 255.0
    if frames.size == 0:
        print("No frames extracted from video, skipping prediction.")
        return 'Error: No frames extracted'
    # Predict frame by frame
    predictions = []
    for frame in frames:
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension for a single frame
        prediction = model.predict(frame)
        predictions.append(prediction[0][0])
    
    avg_prediction = np.mean(predictions)
    label = "FAKE" if avg_prediction > 0.5 else "REAL"
    print(f"Average Prediction: {avg_prediction}, Label: {label}")
    return label

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)