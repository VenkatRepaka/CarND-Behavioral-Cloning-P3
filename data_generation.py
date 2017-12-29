import csv
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

file_path = './data/data/driving_log_dummy_64.csv'
image_data_file_path = './data/data/'
image_shape = (320, 160, 3)
array_shape = [0, 160, 320, 3]
left_steering_correction = 0.25
right_steering_correction = -0.25


def load_data_with_flip():
    x = np.empty(array_shape, np.float32)
    y = np.empty([0], np.float32)
    with open(file_path) as csv_file:
        lines = csv.reader(csv_file, delimiter=',')
        next(lines)
        counter = 0
        for line in lines:
            for k in range(3):
                # image = cv2.imread(image_data_file_path+line[k].strip())
                image = cv2.imread(image_data_file_path+line[k].strip())
                b, g, r = cv2.split(image)
                image = cv2.merge([r, g, b])

                flipped_image = cv2.flip(image, 1, cv2.COLOR_BGR2RGB)
                if k == 0:
                    y = np.append(y, float(line[3]))
                    y = np.append(y, -1 * float(line[3]))
                elif k == 1:
                    y = np.append(y, float(line[3]) + left_steering_correction)
                    y = np.append(y, -1 * (float(line[3]) + left_steering_correction))
                else:
                    y = np.append(y, float(line[3]) + right_steering_correction)
                    y = np.append(y, -1 * (float(line[3]) + right_steering_correction))
                x = np.append(x, [image], axis=0)
                x = np.append(x, [flipped_image], axis=0)
                counter += 1
                if counter % 10 == 0:
                    print('{:2d} images loaded'.format(counter))
    return x, y


def add_random_shadow(image):
    alpha = np.random.uniform(0.2, 0.6)
    h, w, d = image.shape
    zero_or_one = np.random.randint(0, 2)
    if zero_or_one == 0:
        points = np.array([[0, 0], [w - random.randint(0, w), 0], [w - random.randint(0, w), h], [0, h]], np.int32)
    else:
        points = np.array([[w - random.randint(0, w), 0], [w, 0], [w, h], [w - random.randint(0, w), h]], np.int32)

    aug_image = image.copy()
    overlay = image.copy()
    output = overlay.copy()

    # overlay=cv2.rectangle(image, (25, 25), (w-10, h-10), (0,0,0), -1)
    # overlay = cv2.fillConvexPoly(aug_image, points, (0, 0, 0))
    cv2.fillPoly(overlay, [points], (0, 0, 0))
    cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, aug_image)
    return aug_image


def change_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert it to hsv

    h, s, v = cv2.split(hsv)
    v *= np.random.uniform()
    v[v > 255] = 255
    # v = np.array(v, np.uint8)
    final_hsv = cv2.merge((h, s, v))

    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img