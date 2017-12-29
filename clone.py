from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
import data_generation as datagen
from sklearn import model_selection
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


def batch_generator(x, y, batch_size=64):
    while True:
        batch_X = []
        batch_y = []
        for batch in range(0, len(x), batch_size):
            start = batch
            end = batch + batch_size - 1
            if end >= len(x):
                end = len(x) - 1
            for idx in range(start, end):
                check = np.random.randint(0, 2)
                if check == 0:
                    batch_X.append(datagen.add_random_shadow(x[idx]))
                    batch_y.append(y[idx])
                    batch_X.append(datagen.change_brightness(x[idx]))
                    batch_y.append(y[idx])
                else:
                    aug_image = datagen.add_random_shadow(x[idx])
                    aug_image = datagen.change_brightness(aug_image)
                    batch_X.append(aug_image)
                    batch_y.append(y[idx])
        yield(np.array(batch_X), np.array(batch_y))


def display_sample_images(images_to_plot):
    cols = 3
    rows = int(len(images_to_plot)/cols) + len(images_to_plot) % cols
    fig, img_holders = plt.subplots(rows, cols)
    for x in range(len(img_holders)):
        for y in range(len(img_holders[x])):
            if x * cols + y < len(images_to_plot):
                img_holders[x][y].imshow(np.array(images_to_plot[x*cols+y], np.uint8))
                # img_holders[x][y].imshow(images_to_plot[x*cols+y], cmap='gray')
                img_holders[x][y].set_axis_off()
            else:
                img_holders[x][y].axis('off')
    plt.show()


def hist_with_steering_angles():
    return None


def very_simple_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1))
    return model


def simple_model(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=input_shape))
    model.add()
    return model


def nvidia_model():
    return None


def vgg16_model(model_input_shape):
    model = Sequential()

    model.add(Lambda(
        lambda x: (x / 255.0) - 0.5,
        input_shape=model_input_shape
    ))

    model.add(Conv2D(24, filters=(5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, filters=(5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, filters=(5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Dropout(.4))
    model.add(Conv2D(64, filters=(5, 5), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Conv2D(64, filters=(5, 5), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Dropout(.3))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model


batch_size = 128
X, y = datagen.load_data_with_flip()
print('Data load successful')

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2)
print('Train and test split successful')

steps_per_epoch = int(len(X_train)/batch_size)
# validation_steps = int(len(X_valid)/batch_size)
validation_steps = 50
training_generator = batch_generator(X_train, y_train, batch_size=batch_size)
validation_generator = batch_generator(X_valid, y_valid, batch_size=batch_size)

model_input_shape = X_train[0].shape
# model = very_simple_model(model_input_shape)
model = vgg16_model(model_input_shape)
# model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
model.compile(loss='mse', optimizer='adam')
# model.fit(np.array(augmented_images), np.array(steering), validation_split=0.2, shuffle=True, batch_size=128)
model.fit_generator(training_generator, steps_per_epoch = steps_per_epoch, validation_data=validation_generator, validation_steps=validation_steps, epochs=10)
model.save('model.h5')
