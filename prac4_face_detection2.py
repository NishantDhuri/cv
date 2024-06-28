# detect faces and show it on screen
import cv2
import matplotlib.pyplot as plt
# initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# start capturing video from the camera (default camera is usually at index 0)
cap = cv2.VideoCapture(0)
# Capture a single frame
ret, frame = cap.read()
if ret:  # check if the frame was successfully captured
    # convert the captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    # draw rectangles around the detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    # Display the resulting frame with faces highlighted using matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # turn off axis labels
    plt.show()
# Release the capture
cap.release()
print('Number of faces detected: ',len(faces))
