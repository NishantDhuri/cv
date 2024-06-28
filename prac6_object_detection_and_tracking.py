
import cv2
import numpy as np
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
net=cv2.dnn.readNet("D:/Nishant/MSC-IT/cv_new/yolofiles/yolov3.weights", "D:/Nishant/MSC-IT/cv_new/yolofiles/yolov3.cfg")
classes=[]
with open("D:/Nishant/MSC-IT/cv_new/coco.names", "r") as f:
    classes=[line.strip() for line in f.readlines()]
layer_names=net.getLayerNames()
output_layers=[layer_names[i-1] for i in net.getUnconnectedOutLayers()]
cap=cv2.VideoCapture('D:/Nishant/MSC-IT/cv_new/input/video1.mp4')
try:
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        height, width, channels = frame.shape
        blob=cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs=net.forward(output_layers)
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores=detection[5:]
            class_id=np.argmax(scores)

            confidence=scores[class_id]
            if confidence > 0.5:
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        indexes=cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h=boxes[i]
                label=str(classes[class_ids[i]])
                color=(0,255,0) #GReen
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x,y+ 30), cv2.FONT_HERSHEY_PLAIN,1,color,2)
        frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,10))
        plt.imshow(frame_rgb)
        plt.axis('off')
        display(plt.gcf())
        clear_output(wait=True)
        plt.close()
finally:
    cap.release()
    print("Stream ended")