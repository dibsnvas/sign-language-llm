import cv2 as cv
from flask import Flask, render_template, Response, send_from_directory
import copy
from collections import deque, Counter
from imutils.video import WebcamVideoStream
import mediapipe as mp
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import itertools

# Import the required models and utilities
from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the KeyPointClassifier and PointHistoryClassifier models
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Load labels for classifiers
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

# Other initializations
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)
cvFpsCalc = CvFpsCalc(buffer_len=10)

# Start the video stream
stream = WebcamVideoStream(src=0).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

def generate_frames():
    while True:
        frame = stream.read()
        if frame is None:
            continue

        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                logging_csv(0, 0, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                hand_sign_text = keypoint_classifier_labels[hand_sign_id] if 0 <= hand_sign_id < len(keypoint_classifier_labels) else "Unknown"
                finger_gesture_text = point_history_classifier_labels[most_common_fg_id[0][0]]
                debug_image = draw_info_text(debug_image, brect, handedness, hand_sign_text, finger_gesture_text)
        else:
            point_history.append([0, 0])

        # Encoding frame to JPEG
        ret, buffer = cv.imencode('.jpg', debug_image)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Placeholder functions with actual implementation
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

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

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history_classifier_label.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def draw_bounding_rect(is_brect, image, brect):
    if is_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        # Add remaining landmark connections here...
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1] - 22), (brect[2], brect[1]), (0, 0, 0), -1)
    font_path = "DejaVuSans.ttf"  # Ensure this path is correct
    font = ImageFont.truetype(font_path, 22)
    pil_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text += ':' + hand_sign_text
    draw.text((brect[0] + 5, brect[1] - 22), info_text, font=font, fill=(255, 255, 255))
    if finger_gesture_text != "":
        draw.text((brect[0] + 5, brect[1] + 5), "Gesture:" + finger_gesture_text, font=font, fill=(255, 255, 255))
    image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
    return image

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
