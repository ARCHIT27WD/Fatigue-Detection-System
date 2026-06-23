# AI-Based Fatigue Detection System

## Overview
This project is a real-time AI-powered Fatigue Detection System designed to detect drowsiness and fatigue in individuals by monitoring eye closure and yawning patterns using computer vision techniques.

The system analyzes facial landmarks from a live webcam feed and generates alerts when signs of fatigue are detected, helping improve safety in applications such as driver monitoring and workplace alertness systems.

## Features
- Real-time face detection
- Eye Aspect Ratio (EAR) based drowsiness detection
- Yawn detection using facial landmarks
- Live webcam monitoring
- Audio/visual fatigue alerts
- Lightweight and efficient implementation

## Technologies Used
- Python
- OpenCV
- Dlib
- NumPy
- Computer Vision
- Machine Learning Concepts

## Project Architecture

1. Capture live video from webcam
2. Detect face using Haar Cascade
3. Extract facial landmarks using Dlib
4. Calculate Eye Aspect Ratio (EAR)
5. Detect prolonged eye closure
6. Detect yawning activity
7. Trigger fatigue alert when threshold conditions are met

## Files

- `drowsiness_yawn.py` – Main application script
- `haarcascade_frontalface_default.xml` – Face detection model
- `shape_predictor_68_face_landmarks.dat` – Facial landmark predictor

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Fatigue-Detection-System.git
cd Fatigue-Detection-System

pip install opencv-python
pip install dlib
pip install numpy
