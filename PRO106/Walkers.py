from cgitb import grey
from socket import CAN_RTR_FLAG
import cv2

body_classifier = cv2.CascadeClassifier('haarascade_fullbody.sml')
cap = cv2.VideoCapture('walking.avi')

while True:
    
    ret, frame = cap.read()

    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
    bodies = body_classifier.detectMultiScale(grey, 1.2, 3)
    
    if cv2.waitKey(1) == 32: 
        break

cap.release()
cv2.destroyAllWindows()
