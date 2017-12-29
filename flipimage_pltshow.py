import cv2
from matplotlib import pyplot as plt

image = cv2.imread('./tomnjerry_resized.jpg')
b, g, r = cv2.split(image)
image = cv2.merge([r, g, b])
image = cv2.flip(image, 1)
plt.imshow(image)
plt.show()
