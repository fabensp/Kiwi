import cv2
import numpy as np


cap = cv2.imread('Assets/CoopPhoto.jpg', 1)
cap = cv2.resize(cap, (400, 400))

while True:
	ret, frame = cap.read()

	cv2.imshow('frame',frame)

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()


cv2.imshow('Image', cap)
cv2.waitKey(0)
cv2.destroyAllWindows()