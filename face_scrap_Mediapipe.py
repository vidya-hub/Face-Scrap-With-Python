from os import chdir, mkdir
from numpy.linalg import norm
import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.5,)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


def brightness(img):
    if len(img.shape) == 3:

        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)


while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(imgRgb)

    # try:
    if (results.detections) != None:
        for face in (results.detections):
            # mp_drawing.draw_detection(img, face)
            # mp_drawing.draw_detection(img, face)
            face_data = face.location_data.relative_bounding_box
            h, w, c = img.shape
            x, y, width, height = int(face_data.xmin*w), int(face_data.ymin*h), \
                int(face_data.width*w), int(face_data.height*h)
            print(x, y)
            # cv2.circle(img, (x, y), 10, (0, 0, 255),-1)
            cv2.putText(img, "Click c to capture picture", (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
            if (x > 0 and y > 0):
                face_rect = img[y:y+height, x:x+width]
                cv2.imshow("Face", face_rect)
                if(cv2.waitKey(1) == 99):
                    mkdir("no_mask")
                    chdir("no_mask")
                    for i in range(1000):

                        cv2.imwrite(f"Without Mask_{i}.jpg", face_rect,)
                    print("Completed")
                    # print("Cap")
            # cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 255), 1,)
    # except:
    #     print("Face Not Detected")
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
