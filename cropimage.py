import cv2

image = cv2.imread('./tomnjerry_resized.jpg')

# output = image[90:250, 352:477]
output = image[352:477, 90:250]
# 80, 361
# 255, 457
# cv2.imshow('cropped', image)
cv2.imshow('cropped', output)
cv2.waitKey(0)
cv2.destroyWindow("cropped")
