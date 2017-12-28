import csv
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam

images = []
steering = []

left_steering_correction = 0.25
right_steering_correction = -0.25
# model_image_input_shape = (128, 128)
model_image_input_shape = (320, 160)


def write_augmented_images(images_to_write):
    cntr = 0
    for im in images_to_write:
        cv2.imwrite('./write_data/' + str(cntr) + '.jpeg', im)
        cntr += 1


def load_data(file_path, images_data_path):
    with open(file_path) as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        next(lines)
        # counter = 0
        for line in lines:
            # counter += 1
            # if counter > 3:
            #     continue
            for k in range(3):
                image = cv2.cvtColor(cv2.imread(images_data_path + line[k].strip()), cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, dsize=model_image_input_shape)
                images.append(image)
                if k == 0:
                    steering.append(float(line[3]))
                elif k == 1:
                    steering.append(float(line[3]) + left_steering_correction)
                else:
                    steering.append(float(line[3]) + right_steering_correction)
        total_samples = len(images)
        # index_values = np.arange(0, total_samples)
        # return total_samples, index_values
        return total_samples


def image_indices_to_augment(no_of_images, ratio=50):
    no_to_augment = int(no_of_images * ratio / 100)
    sampling_indices = np.arange(0, no_of_images)
    augment_image_indices = np.random.choice(sampling_indices, no_to_augment, replace=False)
    return augment_image_indices
    # return sampling_indices


def add_shadow(images_to_shadow, ratio=50):
    alpha = 0.3
    augmented_images = []
    steering_augment = []
    augment_image_indices = image_indices_to_augment(len(images_to_shadow), ratio)
    for idx in augment_image_indices:
        aug_image = images_to_shadow[idx].copy()
        steering_augment.append(steering[idx])

        h, w, d = aug_image.shape
        zero_or_one = np.random.randint(0, 2)
        if zero_or_one == 0:
            points = np.array([[0, 0], [w - random.randint(0, w), 0], [w - random.randint(0, w), h], [0, h]], np.int32)
        else:
            points = np.array([[w - random.randint(0, w), 0], [w, 0], [w, h], [w - random.randint(0, w), h]], np.int32)

        overlay = aug_image.copy()
        output = aug_image.copy()

        # overlay=cv2.rectangle(image, (25, 25), (w-10, h-10), (0,0,0), -1)
        # overlay = cv2.fillConvexPoly(aug_image, points, (0, 0, 0))
        cv2.fillPoly(overlay, [points], (0, 0, 0))
        cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, aug_image)
        augmented_images.append(aug_image)
        # cv2.imshow('overlayed', aug_image)
        # cv2.waitKey(0)
        # cv2.destroyWindow("overlayed")
    return augmented_images, steering_augment


def crop_images(images_to_crop):
    cropped_images = [img[70:120, 0:320] for img in images_to_crop]
    return cropped_images


def generate_mirror_images():
    flipped_images = [cv2.flip(image, 1) for image in images]
    flipped_steering = [steering_angle*-1 for steering_angle in steering]
    return flipped_images, flipped_steering


def augment_data(ratio=50, plot_data=False, write_augmented_images=False):
    print('Start data augmentation')
    flipped_images, flipped_steering = generate_mirror_images()
    print('Mirror image generation complete')
    original_flipped_images = flipped_images + images
    steering.extend(flipped_steering)
    shadowed_images, shadowed_steering = add_shadow(original_flipped_images, ratio)
    print('Shadowing images complete')
    cropped_images = crop_images(shadowed_images)
    print('Cropping images done')
    print(cropped_images[0].shape)
    if write_augmented_images:
        write_augmented_images(cropped_images)
    if plot_data:
        if len(cropped_images) >= 6:
            random_images_idx = np.random.choice(len(cropped_images), 12)
            plot_images = []
            for idx in random_images_idx:
                plot_images.append(cropped_images[idx])
            display_sample_images(plot_images)
    return cropped_images


def display_sample_images(images_to_plot):
    cols = 3
    rows = int(len(images_to_plot)/cols) + len(images_to_plot) % cols
    fig, img_holders = plt.subplots(rows, cols)
    for x in range(len(img_holders)):
        for y in range(len(img_holders[x])):
            if x * cols + y < len(images_to_plot):
                img_holders[x][y].imshow(images_to_plot[x*cols+y])
                # img_holders[x][y].imshow(images_to_plot[x*cols+y], cmap='gray')
                img_holders[x][y].set_axis_off()
            else:
                img_holders[x][y].axis('off')
    plt.show()
    return None


def hist_with_steering_angles():
    return None


def very_simple_model():
    model = Sequential()
    model.add(Flatten(input_shape=(50, 320, 3)))
    model.add(Dense(1))
    return model


def simple_model():
    return None


def nvidia_model():
    return None


def vgg16_model():
    return None


initial_sample = load_data('./data/data/driving_log.csv', './data/data/')
print('Data load successful')
augmented_images = augment_data(100, plot_data=False)
# plt.imshow(cv2.cvtColor(cv2.imread('./data/data/IMG/center_2016_12_01_13_30_48_287.jpg'), cv2.COLOR_BGR2RGB))
# plt.show()

model = very_simple_model()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
model.fit(np.array(augmented_images), np.array(steering), validation_split=0.2, shuffle=True)
model.save('model.h5')
