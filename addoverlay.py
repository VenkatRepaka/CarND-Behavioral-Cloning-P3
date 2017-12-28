import cv2
import numpy as np
import random

image = cv2.imread('./tomnjerry.jpg')
h, w, d = image.shape
h, w = int(h/4), int(w/4)
image = cv2.resize(image, dsize=(w, h))
alpha = 0.5
overlay_color_1 = (0, 0, 255)
overlay_color_2 = (255, 0, 0)
# for alpha in np.arange(0.0, 1.1, 0.1):
overlay = image.copy()
output = image.copy()

#                           Top     Top-Right                      Bottom-Right                   Bottom-Left
points1 = points = np.array([[0, 0], [w - random.randint(0, w), 0], [w - random.randint(0, w), h], [0, h]], np.int32)
points2 = points = np.array([[w - random.randint(0, w), 0], [w, 0], [w, h], [w - random.randint(0, w), h]], np.int32)
print(points2)
# cv2.rectangle(overlay, (113, 356), (200, 440), overlay_color, -1)
# cv2.fillConvexPoly(overlay, points, overlay_color)
cv2.fillPoly(overlay, [points1], overlay_color_1)
cv2.fillPoly(overlay, [points2], overlay_color_2)
cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)

cv2.imshow('overlayed', output)
cv2.waitKey(0)
cv2.destroyWindow("overlayed")