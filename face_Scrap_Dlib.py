import math
import cv2
import numpy as np
import dlib
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    print(faces)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        print(x1, y1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.imshow("original", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
