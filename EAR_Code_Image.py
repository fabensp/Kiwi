import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist

#Global Configuration Variables
FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  # path to dlib's pre-trained facial landmark predictor
MINIMUM_EAR = 0.05    # Minimum EAR for both the eyes to mark the eyes as open


#Initializations
faceDetector = dlib.get_frontal_face_detector()     # dlib's HOG based face detector
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)  # dlib's landmark finder/predcitor inside detected face

# Finding landmark id for left and right eyes
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear


image = cv2.imread("9.jpg")
image = imutils.resize(image, width=800)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceDetector(grayImage, 0)

for face in faces:
    faceLandmarks = landmarkFinder(grayImage, face)
    faceLandmarks = face_utils.shape_to_np(faceLandmarks)

    print(faceLandmarks[leftEyeStart])
    print(faceLandmarks[leftEyeEnd])

    leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
    rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)

    for i in range(len(faceLandmarks)):
        print(i)
        print(faceLandmarks[i])

    cv2.drawContours(image, [leftEyeHull], -1, (255, 0, 0), 2)
    cv2.drawContours(image, [rightEyeHull], -1, (255, 0, 0), 2)

    cv2.putText(image, "EAR: {}".format(round(ear, 3)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if ear < MINIMUM_EAR:
        cv2.putText(image, "Drowsiness", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Frame", image)
cv2.waitKey(0)