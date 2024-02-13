import cv2
import dlib

# load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image = cv2.imread("images/3.jpg")
image = cv2.resize(image, (600, 500))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect the faces
rects = detector(gray)

# go through the face bounding boxes
for rect in rects:
    # extract the coordinates of the bounding box
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # apply the shape predictor to the face ROI
    shape = predictor(gray, rect)

	for n in range(0, 68):
        x = shape.part(n).x
    	y = shape.part(n).y
        cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
        
cv2.imshow("Image", image)
cv2.waitKey(0)