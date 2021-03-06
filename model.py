import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, Convolution2D
from keras.optimizers import Adam

import data_generation as dg
from sklearn import model_selection
import pandas as pd
from sklearn.utils import shuffle


def load_data(image_data_path):
    data = pd.read_csv(image_data_path)
    x_data = np.array([])
    y_data = np.array([])
    x_data = np.append(x_data, data['center'])
    steering_center = data['steering']
    x_data = np.append(x_data, data['left'])
    steering_left = data['steering'] + left_steering_correction
    x_data = np.append(x_data, data['right'])
    steering_right = data['steering'] + right_steering_correction
    y_data = np.append(y_data, steering_center)
    y_data = np.append(y_data, steering_left)
    y_data = np.append(y_data, steering_right)
    x_data, y_data = shuffle(x_data, y_data)
    x_train_data, x_valid_data, y_train_data, y_valid_data = model_selection.train_test_split(np.array(x_data),
                                                                                              np.array(y_data),
                                                                                              test_size=0.2)
    return x_train_data, x_valid_data, y_train_data, y_valid_data


def very_simple_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1))
    return model


def nvidia_model(model_input_shape):
    model = Sequential()
    model.add(Lambda(
        lambda x: (x / 255.0) - 0.5,
        input_shape=model_input_shape
    ))
    # Color space conversion layer
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Flatten())

    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model


left_steering_correction = 0.25
right_steering_correction = -0.25
data_dir = "./data/"
keep_prob = 0.5
learning_rate = 0.0001
batch_size = 128
num_train_images = 19200
num_val_images = 4820
# num_train_images = 100
# num_val_images = 50
x_train, x_valid, y_train, y_valid = load_data(data_dir+'driving_log.csv')
# steps_per_epoch = 1
# validation_steps = 1
train_data_generator = dg.train_data_generator(data_dir, x_train, y_train, batch_size)
valid_data_generator = dg.train_data_generator(data_dir, x_valid, y_valid, batch_size, augment=False)
# x_train_batch, x_train_batch = dg.train_data_generator(data_dir, x_train, y_train, batch_size=128)
# x_valid_batch, x_valid_batch = dg.train_data_generator(data_dir, x_valid, y_valid, batch_size=128, augment=False)
exec_model = nvidia_model((64, 64, 3))
exec_model.summary()
exec_model.compile(optimizer=Adam(learning_rate), loss="mse")
# exec_model.fit_generator(train_data_generator,
#                          steps_per_epoch=steps_per_epoch,
#                          validation_data=valid_data_generator,
#                          validation_steps=validation_steps,
#                          epochs=1,
#                          verbose=1)
exec_model.fit_generator(train_data_generator,
                         samples_per_epoch=num_train_images,
                         nb_epoch=30,
                         validation_data=valid_data_generator,
                         nb_val_samples=num_val_images,
                         verbose=1)
exec_model.save('model.h5')
