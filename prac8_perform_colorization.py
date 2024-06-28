
import numpy as np
import cv2
import matplotlib.pyplot as plt

net = cv2.dnn.readNetFromCaffe('D:/Nishant/MSC-IT/cv_new/colorization_deploy_v2.prototxt','D:/Nishant/MSC-IT/cv_new/colorization_release_v2.caffemodel')

pts_in_hull = np.load('D:/Nishant/MSC-IT/cv_new/pts_in_hull.npy' , allow_pickle=True)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts_in_hull.astype(np.float32)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, np.float32)]

image = cv2.imread('D:/Nishant/MSC-IT/cv_new/input/lana.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

normalized_image = gray_image.astype('float32') / 255.0

lab_image = cv2.cvtColor(normalized_image, cv2.COLOR_RGB2Lab)

resized_1_channel = cv2.resize(lab_image[:, :, 0], (224, 224))
resized_1_channel -= 50

net.setInput(cv2.dnn.blobFromImage(resized_1_channel))
pred = net.forward()[0, :, :, :].transpose((1, 2, 0))

pred_resized = cv2.resize(pred, (image.shape[1], image.shape[0]))

colorized_image = np.concatenate((lab_image[:, :, 0][:, :, np.newaxis], pred_resized), axis=2)

colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_Lab2BGR)

colorized_image = np.clip(colorized_image, 0, 1)

colorized_image = (255 * colorized_image).astype('uint8')
cv2.imwrite('path_to_output/colorized_image.jpg', colorized_image)

colorized_image_rgb = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 8))
plt.imshow(colorized_image_rgb)
plt.axis('off')
plt.show()