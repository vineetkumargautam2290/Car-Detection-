import os
import numpy as np
import pandas as pd
import cv2
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')
cap = cv2.VideoCapture('cars.avi')

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray, 1.3,3)
    for (x,y, w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255),3)
        cv2.imshow("Car is running on the highway", frame)

    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()