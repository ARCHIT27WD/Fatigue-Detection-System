# Import necessary libraries
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Alarm function for voice alerts
def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        engine.say(msg)
        engine.runAndWait()

    if alarm_status2:
        saying = True
        engine.say(msg)
        engine.runAndWait()
        saying = False

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Get final EAR from both eyes
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

# Calculate mouth opening distance
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    return abs(top_mean[1] - low_mean[1])

# Argument parser for webcam index
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# Constants for thresholds
EYE_AR_THRESH = 0.23  # Adjusted for better blink detection
EYE_AR_CONSEC_FRAMES = 5
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
BLINK_COUNT = 0
START_TIME = time.time()

# Load face detector and landmark predictor
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Start video stream
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # ----------------- Ambient Light Monitoring -----------------
        brightness = gray.mean()
        LOW_LIGHT_THRESH = 60  # You can adjust this threshold based on your setup

        if brightness < LOW_LIGHT_THRESH:
            # Ensure the message is visible at the top of the screen
            cv2.putText(frame, "Low Light Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)



        
        # Calculate EAR and detect blinks
        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        # Draw contours on eyes and lips
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Blink Detection
        BLINK_THRESH_FRAMES = 2  # Minimum frames to register a blink

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            blink_detected = False
        else:
            if COUNTER >= BLINK_THRESH_FRAMES:
                BLINK_COUNT += 1
                blink_detected = True
                print("Blink Detected")
            COUNTER = 0

        # Drowsiness Detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES or ear < 0.15:  # <- Additional hard threshold
                if not alarm_status:
                    alarm_status = True
                    t = Thread(target=alarm, args=('Wake up sir!',))
                    t.daemon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False


        # Yawn Detection
        if distance > YAWN_THRESH:
            cv2.putText(frame, "Yawn Alert", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not alarm_status2 and not saying:
                alarm_status2 = True
                t = Thread(target=alarm, args=('Take some fresh air sir!',))
                t.daemon = True
                t.start()
        else:
            alarm_status2 = False

        # Calculate blinks per minute
        elapsed_time = time.time() - START_TIME
        if elapsed_time >= 60:
            blinks_per_minute = BLINK_COUNT
            BLINK_COUNT = 0
            START_TIME = time.time()
        else:
            blinks_per_minute = BLINK_COUNT * (60 / elapsed_time)


        # ----------------- Sleepiness Level Estimation -----------------
        # Normalize blink score (more blinks = more sleepiness)
        blink_score = min(BLINK_COUNT / 10.0, 3.0)  # cap at 3

        # Normalize yawn score (more yawns = more sleepiness)
        yawn_score = 1.0 if distance > YAWN_THRESH else 0.0

        # Normalize EAR score (low EAR = more sleepiness)
        ear_score = 1.0 if ear < EYE_AR_THRESH else 0.0

        # Weights (tweak for better sensitivity)
        blink_weight = 1.5
        yawn_weight = 2.0
        ear_weight = 2.5

        # Total sleepiness score
        sleepiness_score = (blink_weight * blink_score) + (yawn_weight * yawn_score) + (ear_weight * ear_score)

        # Sleepiness Level classification
        if sleepiness_score < 3:
            sleepiness_level = "Low"
        elif sleepiness_score < 6:
            sleepiness_level = "Medium"
        else:
            sleepiness_level = "High"

        # Display Sleepiness Level
        cv2.putText(frame, f"Sleepiness: {sleepiness_level}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 255) if sleepiness_level == "High" else (0, 255, 255) if sleepiness_level == "Medium" else (0, 255, 0), 2)





        # Display EAR, Yawn distance, and Blink count
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Blinks: {BLINK_COUNT}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"BPM: {blinks_per_minute:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


   


    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit on 'q'
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()