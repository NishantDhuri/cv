#!pip install opencv-python
#!pip install matplotlib 'D:/Nishant/MSC-IT/cv_new/input/video1.mp4'
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Initialize the Haar Cascade pedestrian detection model
pedestrians_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
# Load the video
video_path = 'D:/Nishant/MSC-IT/cv_new/input/video1.mp4'
cap = cv2.VideoCapture(video_path)
# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video")
    exit()
# Function to detect pedestrians and draw bounding boxes
def detect_pedestrians(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect pedestrians in the grayscale frame
    pedestrians = pedestrians_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    # Draw rectangle around the detected pedestrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    return frame
# Create a subplot for displaying the video frame
fig, ax = plt.subplots()
# Function to update the animation
def update(frame):
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        ani.event_source.stop()
        return
    # Detect pedestrians and draw bounding boxes
    frame_with_pedestrians = detect_pedestrians(frame)
    # Display the frame
    ax.clear()
    ax.imshow(cv2.cvtColor(frame_with_pedestrians, cv2.COLOR_BGR2RGB))
    ax.axis('off')   # Turn off axis labels
# Create the animation
ani = animation.FuncAnimation(fig, update, interval=50)
plt.show()
# Release the video capture object
cap.release()
