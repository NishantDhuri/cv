!pip install dlib-19.22.99-cp38-cp38-win_amd64.whl
!pip install face_recognition
import face_recognition
from matplotlib.patchesimport Rectangle
import matplotlib.pyplot as plt
known_image = face_recognition.load_image_file('ryan.jpg')
known_face_encoding = face_recognition.face_encodings(known_image)[0]
unknown_image = face_recognition.load_image_file('test_ryan.jpg')
unknown_face_locations = face_recognition.face_locations(unknown_image)
unknown_face_encodings = face_recognition.face_encodings(unknown_image,
unknown_face_locations)
213
unknown_image_rgb = unknown_image[:, :, ::-1]
fig,ax= plt.subplots(figsize=(8,6))
ax.imshow(unknown_image_rgb)
for (top, right, bottom, left), unknown_face_encoding in zip(unknown_face_locations,
unknown_face_encodings):
results=face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)
if results[0]:
name="Known-Ryan"
else:
name="Unknown"
ax.add_patch(Rectangle((left, top), right - left, bottom-top, fill=False, color='green',
linewidth=2))
ax.text(left+6, bottom+25, name, color='white', fontsize=12)
plt.axis('off')
