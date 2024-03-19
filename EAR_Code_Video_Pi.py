import cv2
import dlib
import imutils
import time
from imutils import face_utils
from scipy.spatial import distance as dist
import threading
from picamera2 import Picamera2
import pigpio

pi = pigpio.pi()

#Global Configuration Variables
FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  # path to dlib's pre-trained facial landmark predictor

#Initializations
faceDetector = dlib.get_frontal_face_detector()     # dlib's HOG based face detector
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)  # dlib's landmark finder/predcitor inside detected face

# Finding landmark id for left and right eyes
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# timer variable used to keep track of elasped time
ear = 0
sidelook_ratio = 0

pt = time.time
t = time.time

MINIMUM_EAR = 0.15

def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={'size': (1920, 1080), 'format': 'XRGB8888'})
picam2.configure(config)

picam2.start()

while True:
    frame = picam2.capture_array("main")

    scale = 50

    #get the webcam size
    height, width, channels = frame.shape

    #prepare the crop
    centerX,centerY=int(height/2),int(width/2)
    radiusX,radiusY= int(scale*height/100),int(scale*width/100)

    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    frame_small = frame[minY:maxY, minX:maxY]

    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    faces = faceDetector(gray)

    if not faces:
        ear = -1
        sidelook_ratio = -1

    for face in faces:
        faceLandmarks = landmarkFinder(gray, face)
        faceLandmarks = face_utils.shape_to_np(faceLandmarks)

        leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
        rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        sidelook_ratio = p1_minus_p4 = dist.euclidean(leftEye[0], leftEye[3])/dist.euclidean(rightEye[0], rightEye[3])

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

    dt = t - pt
    print(dt)

    pt = t
    t = time.time

    if ear != -1 and (ear < MINIMUM_EAR or sidelook_ratio > 1.2 or sidelook_ratio < 0.8):
        pi.write(17, 1)
    else:
        pi.write(17, 0)

    if cv2.waitKey(1) == ord('q'):
        break

