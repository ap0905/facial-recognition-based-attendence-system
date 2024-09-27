from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load the pre-trained data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Define column names for the CSV file
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    # Draw the instruction messages on the frame
    instruction_text1 = "Press 'o' for attendance"
    instruction_text2 = "Press 'q' to exit"
    
    font_scale = 2
    font_thickness = 4
    text_size1, _ = cv2.getTextSize(instruction_text1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_size2, _ = cv2.getTextSize(instruction_text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    text_x1 = (frame.shape[1] - text_size1[0]) // 2
    text_y1 = 50
    text_x2 = (frame.shape[1] - text_size2[0]) // 2
    text_y2 = text_y1 + text_size1[1] + 20
    
    cv2.putText(frame, instruction_text1, (text_x1, text_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
    cv2.putText(frame, instruction_text2, (text_x2, text_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        csv_file = "Attendance/Attendance_" + date + ".csv"
        exist = os.path.isfile(csv_file)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        attendance = [str(output[0]), str(timestamp)]
    
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    
    if k == ord('o'):
        # Write attendance to CSV file
        if exist:
            with open(csv_file, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open(csv_file, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
    
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

