import cv2

image = cv2.imread('./tomnjerry.jpg')
h, w, d = image.shape
h, w = int(h/4), int(w/4)
image = cv2.resize(image, dsize=(w, h))
flipped = cv2.flip(image, 1)
cv2.imshow('flipped', flipped)
cv2.waitKey(0)
cv2.destroyWindow("flipped")
