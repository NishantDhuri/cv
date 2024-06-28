import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_keypoints_and_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors
def match_keypoints(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
def draw_matches(img1, kp1, img2, kp2, matches):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
def reconstruct_3d_points(K, kp1, kp2, matches, E):
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    # Triangulate points
    P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    points_4d_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
    points_3d = points_4d[:3, :].T
    return points_3d
# Load two images of a scene
img1 = cv2.imread('D:/Nishant/MSC-IT/cv_new/lana.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('D:/Nishant/MSC-IT/cv_new/pana.jpg', cv2.IMREAD_GRAYSCALE)

# Placeholder values for the camera matrix K (Students you should change it based on yourcamera parameters.
fx = 1000 # Focal length in pixels
fy = 1000 # Focal length in pixels
cx = img1.shape[1] / 2 # Optical center X coordinate, assuming center of the image
cy = img1.shape[0] / 2 # Optical center Y coordinate, assuming center of the image
K = np.array([[fx, 0, cx],
[0, fy, cy],
[0, 0, 1]], dtype=float)
# Extract keypoints and descriptors
kp1, desc1 = extract_keypoints_and_descriptors(img1)
kp2, desc2 = extract_keypoints_and_descriptors(img2)
# Match keypoints
matches = match_keypoints(desc1, desc2)
# Estimate the essential matrix
points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
E = K.T @ F @ K
# Reconstruct 3D points
points_3d = reconstruct_3d_points(K, kp1, kp2, matches, E)
# Visualize 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
plt.show()

# Display matched keypoints
matched_img = draw_matches(img1, kp1, img2, kp2, matches)
plt.imshow(matched_img)
plt.show()
