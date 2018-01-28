import csv
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

file_path = './data/data/driving_log_dummy_16.csv'
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


def load_data_with_flip_numpy():
    # x = np.empty(array_shape, np.float32)
    # y = np.empty([0], np.float32)
    x = []
    y = []
    with open(file_path) as csv_file:
        lines = csv.reader(csv_file, delimiter=',')
        next(lines)
        counter = 0
        for line in lines:
            for k in range(3):
                image = cv2.imread(image_data_file_path + line[k].strip())
                b, g, r = cv2.split(image)
                image = cv2.merge([r, g, b])
                #
                # flipped_image = cv2.flip(image, 1, cv2.COLOR_BGR2RGB)
                if k == 0:
                    y.append(float(line[3]))
                    # y.append(-1 * float(line[3]))
                elif k == 1:
                    y.append(float(line[3]) + left_steering_correction)
                    # y.append(-1 * (float(line[3]) + left_steering_correction))
                else:
                    y.append(float(line[3]) + right_steering_correction)
                    # y.append(-1 * (float(line[3]) + right_steering_correction))
                x.append(image)
                # x.append(flipped_image)
                counter += 1
                if counter % 10 == 0:
                    print('{:2d} images loaded'.format(counter))
    return np.array(x), np.array(y)


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
    z = v*np.random.uniform(0.2, 0.8)
    z[z > 255] = 255
    v = np.array(z, np.uint8)
    final_hsv = cv2.merge((h, s, v))

    result_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return result_image


def mask_image(image):
    # Applies an image mask.
    # region_of_interest(image, vertices):
    rows, cols, ch = image.shape
    ax = int(cols * (np.random.uniform(-0.5, 0.5)))
    bx = int(ax + cols * np.random.uniform(-0.5, 0.5))
    cx = int(np.random.uniform(0, 80))
    dx = int(cols - cx)
    p = (np.random.uniform(-0.5, 0.5))
    # vertices = np.array([[(p*cols,rows),(ax,int(p*rows)), (bx, int(p*rows)), (cols*(1+p),rows)]], dtype=np.int32)
    vertices = np.array([[(dx, rows), (ax, int(p * rows)), (bx, int(p * rows)), (cx, rows)]], dtype=np.int32)
    shadow = np.random.randint(1, 200)
    mask = np.full_like(image, shadow)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def flip_image(image):
    return cv2.flip(image, 1)


def crop_relevant_image(image, top_percent=0.35, bottom_percent=0.1):
    rows, cols, ch = image.shape
    image = image[int(rows * top_percent):int(rows - rows * bottom_percent), :]
    return image


def random_shear(image, steering_angle, shear_range=100):
    # Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx/(rows / 2) * 360/(2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering
    return image, steering_angle


def horizontal_crop(image, angle):
    # https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2
    # random pan the camera from left to right in a small pixel shift
    # shift between -24 to 24 pixels and compensate the steering angle
    rows, cols, ch = image.shape
    width = int(cols * 0.8)
    x_var = int(np.random.uniform(-24, 24))
    crop_img = image[0:rows, int(cols / 2 - width / 2 + x_var):int(cols / 2 + width / 2 + x_var)]
    angle_factor = 0.002  # degree per each shifted pixel
    # Since image is 320 px wide. angle_factor will be 1/320 - 0.003125
    angle_factor = 0.0031  # degree per each shifted pixel
    adj_angle = angle + angle_factor * x_var
    return crop_img, adj_angle


def random_crop(image, steering=0.0, tx_lower=-20, tx_upper=20, ty_lower=-2, ty_upper=2, rand=True):
    # we will randomly crop subsections of the image and use them as our data set.
    # also the input to the network will need to be cropped, but of course not randomly and centered.
    shape = image.shape
    col_start, col_end = abs(tx_lower), shape[1] - tx_upper
    horizon = 60
    bonnet = 136
    if rand:
        tx = np.random.randint(tx_lower, tx_upper + 1)
        ty = np.random.randint(ty_lower, ty_upper + 1)
    else:
        tx, ty = 0, 0

    #    print('tx = ',tx,'ty = ',ty)
    random_crop = image[horizon + ty:bonnet + ty, col_start + tx:col_end + tx, :]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(random_crop)
    ax2.set_title('Cropped Image', fontsize=30)
    plt.waitforbuttonpress()
    plt.close()
    image = cv2.resize(random_crop, (64, 64), cv2.INTER_AREA)
    # the steering variable needs to be updated to counteract the shift
    if tx_lower != tx_upper:
        dsteering = -tx / (tx_upper - tx_lower) / 3.0
    else:
        dsteering = 0
    steering += dsteering

    return image, steering


def resize_image(image, res_dim):
    return cv2.resize(image, res_dim)


def random_gamma(image):
    # Source: http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def pipeline(image, steering_angle):
    flip = np.random.choice([0, 1])
    shadow = np.random.choice([0, 1])
    brightness = np.random.choice([0, 1])
    help_recenter = np.random.choice([0, 1])
    shear = np.random.choice([0, 1])
    processed_image = np.copy(image)
    if flip == 1:
        processed_image = flip_image(processed_image)
        steering_angle = -1 * steering_angle
    if brightness == 1:
        processed_image = change_brightness(processed_image)
    # processed_image = random_gamma(processed_image)
    if shadow == 1:
        processed_image = add_random_shadow(processed_image)
    processed_image = crop_relevant_image(processed_image)
    if help_recenter == 1:
        processed_image, steering_angle = horizontal_crop(processed_image, steering_angle)
    if shear == 1:
        processed_image, steering_angle = random_shear(processed_image, steering_angle)
    return processed_image, steering_angle


def pipeline_show_images(image, steering_angle):
    flip = np.random.choice([0, 1])
    shear = np.random.choice([0, 1])
    processed_image = np.copy(image)
    images = []
    if flip == 1:
        processed_image = flip_image(processed_image)
        steering_angle = -1 * steering_angle
    images.append(np.copy(processed_image))
    processed_image = change_brightness(processed_image)
    images.append(np.copy(processed_image))
    # processed_image = random_gamma(processed_image)
    images.append(np.copy(processed_image))
    # processed_image = mask_image(processed_image)
    processed_image = add_random_shadow(processed_image)
    images.append(np.copy(processed_image))
    processed_image = crop_relevant_image(processed_image)
    images.append(np.copy(processed_image))
    processed_image, steering_angle = horizontal_crop(processed_image, steering_angle)
    images.append(np.copy(processed_image))
    if shear == 1:
        processed_image, steering_angle = random_shear(processed_image, steering_angle)
    images.append(np.copy(processed_image))
    show_images(image, images)
    return processed_image, steering_angle


def show_images(image, images):
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(32, 16))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=30)

    ax2.imshow(images[0])
    ax2.set_title('Flipped Image', fontsize=30)

    ax3.imshow(images[1])
    ax3.set_title('Brightness Mod Image', fontsize=30)

    ax4.imshow(images[2])
    ax4.set_title('Gaama Image', fontsize=30)

    ax5.imshow(images[3])
    ax5.set_title('Masked Image', fontsize=30)

    ax6.imshow(images[4])
    ax6.set_title('Relevant Ahead', fontsize=30)

    ax7.imshow(images[5])
    ax7.set_title('Horizontal Crop', fontsize=30)

    ax8.imshow(images[6])
    ax8.set_title('Sheared Image', fontsize=30)

    plt.waitforbuttonpress()
    plt.close()


def train_data_generator(image_dir, x_data, y_data, batch_size=64, augment=True):
    end = 0
    while True:
        to_process_images = []
        to_process_steering = []
        start = end
        end = start+batch_size
        to_process_x = x_data[start:end]
        to_process_y = y_data[start:end]
        if len(to_process_x) < 64:
            start = 0
            end = 64 - len(to_process_x)
            to_process_x = np.append(to_process_x, x_data[start:end])
            to_process_y = np.append(to_process_y, y_data[start:end])
        for image_path, steering in zip(to_process_x, to_process_y):
            if augment:
                new_image, new_steering = pipeline(mpimg.imread(image_dir+image_path.strip()), steering)
                to_process_images.append(cv2.resize(new_image, (64, 64)))
                to_process_steering.append(new_steering)
            else:
                to_process_images.append(cv2.resize(mpimg.imread(image_dir+image_path.strip()), (64, 64)))
                to_process_steering.append(steering)
        # return to_process_images, to_process_steering
        yield (np.array(to_process_images), np.array(to_process_steering))
        # yield (np.array(to_process_x), np.array(to_process_y))


# img = mpimg.imread("./data/IMG/center_2016_12_01_13_30_48_287.jpg")
# masked_image = mask_image(img)
# res_image, steering = pipeline(img, 0)
# res_image, steering = pipeline_show_images(img, 0)
# res_image = cv2.resize(res_image, (64, 64))
# res_image, steering_angle = random_crop(img)
# plt.imshow(res_image)
# plt.waitforbuttonpress()
# plt.close()
# copy_image = np.copy(img)
# f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(32, 16))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=30)
# ax2.imshow(flip_image(img))
# ax2.set_title('Flipped Image', fontsize=30)
# ax3.imshow(add_random_shadow(img))
# ax3.set_title('Masked Image', fontsize=30)
# ax4.imshow(change_brightness(img))
# ax4.set_title('Brightness Mod Image', fontsize=30)
# ax5.imshow(crop_relevant_image(img))
# ax5.set_title('Look Ahead', fontsize=30)
# ax6.imshow(random_shear(copy_image, 1))
# ax6.set_title('Sheared', fontsize=30)
# plt.waitforbuttonpress()
# plt.close()
