#!pip install opencv-python

# detect face from input image and save it on the disk.
import cv2
# Load the pre-trained Haar Cascade model for face detection 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# load the image where you want to detect face
image_path = 'D:/Nishant/MSC-IT/cv_new/input/lana.jpg'  # path to your image
image = cv2.imread(image_path)
# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
# draw rectangles around each face
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
# save the image with faces highlighted 
output_path = 'D:/Nishant/MSC-IT/cv_new/input/faces_detected.jpg'   # corrected file extension
cv2.imwrite(output_path, image)
print(f"Faces Detected: {len(faces)}. Output saved to {output_path}")
