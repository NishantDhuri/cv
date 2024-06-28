#!pip install opencv-python
#!pip install matplotlib 'D:/Nishant/MSC-IT/cv_new/input/video1.mp4'
import cv2
import matplotlib.pyplot as plt

pedestrian_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_fullbody.xml')

video_path='D:/Nishant/MSC-IT/cv_new/input/video1.mp4'
cap=cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open the video")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    pedestrians=pedestrian_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

cap.release()
