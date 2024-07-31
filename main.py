from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from model import KeyPointClassifier, PointHistoryClassifier
import csv
import copy
import itertools
from collections import Counter, deque

app = Flask(__name__)

# Initialize Mediapipe and classifiers
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Process image with Mediapipe
        image_rgb = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(image_rgb, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                point_history.append(landmark_list[8] if 0 <= hand_sign_id < len(keypoint_classifier_labels) else [0, 0])
                finger_gesture_id = 0
                if len(pre_process_point_history(image_rgb, point_history)) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_process_point_history(image_rgb, point_history))
                
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()
                
                # Return predictions
                hand_sign_text = keypoint_classifier_labels[hand_sign_id] if 0 <= hand_sign_id < len(keypoint_classifier_labels) else "Unknown"
                finger_gesture_text = point_history_classifier_labels[most_common_fg_id[0][0]] if most_common_fg_id else "Unknown"
                return jsonify({
                    "hand_sign": hand_sign_text,
                    "gesture": finger_gesture_text
                })
        else:
            return jsonify({"error": "No hands detected"})
    except Exception as e:
        return jsonify({"error": str(e)})

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    max_value = image_width
    def normalize_(n):
        return n / max_value
    temp_point_history = list(map(normalize_, temp_point_history))
    return temp_point_history

if __name__ == "__main__":
    app.run(debug=True)
