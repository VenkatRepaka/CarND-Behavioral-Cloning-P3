# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


### Submission
Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:

1. model.py containing the script to create and train the model
2. data_generation.py the script for data augmentation and batch generator
3. drive.py for driving the car in autonomous mode
4. model.h5 containing a trained convolution neural network


### Preprocessing Data

I have used sample data provided by udacity. There are more than 24000 images in data set.
Below are the augmentation steps that I have used.

- Steering adjustment - Left and right camera data can be used in the training. An adjustment of 0.25 and -0.25 of steering for left and right cameras respectively
```
steering_left = data['steering'] + 0.25
steering_right = data['steering'] - 0.25
```

- Flip Image - Flip the image and change the steering angle. This will help increasing test data
```
cv2.flip(image, 1)
```

- Change Brightness - Will be helpful in dark/brighter conditions
```
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert it to hsv

h, s, v = cv2.split(hsv)
z = v*np.random.uniform(0.2, 0.8)
z[z > 255] = 255
v = np.array(z, np.uint8)
final_hsv = cv2.merge((h, s, v))

result_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
```

- Add random shadow
```
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
```

- Crop relevant portion of the image - I have chosen to ignore top 35% and bottom 10%
```
rows, cols, ch = image.shape
image = image[int(rows * top_percent):int(rows - rows * bottom_percent), :]
```

- Random Shear - I have seen many blogs from medium using random shear.
![One reference](https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2)
```
rows, cols, ch = image.shape
dx = np.random.randint(-shear_range, shear_range + 1)
random_point = [cols / 2 + dx, rows / 2]
pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
pts2 = np.float32([[0, rows], [cols, rows], random_point])
dsteering = dx/(rows / 2) * 360/(2 * np.pi * 25.0) / 6.0
M = cv2.getAffineTransform(pts1, pts2)
image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
steering_angle += dsteering
```
And the steering is also adjusted with the amount of shear

- Pipeline used - Randomly applied one of the above steps to create an image that will be used to train the model
```
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
```

Below are few sample images applying the above pipeline
![fig1](https://github.com/VenkatRepaka/CarND-Behavioral-Cloning-P3/blob/master/readme_data/figure_1.png)
![fig2](https://github.com/VenkatRepaka/CarND-Behavioral-Cloning-P3/blob/master/readme_data/figure_2.png)

### Model Architecture
Most of the students and the blogs on medium have preferred NVIDIA model. So I chose it here.
I have used almost same as mentioned in the [paper](https://arxiv.org/abs/1604.07316)
```
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
```
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
color_conv (Convolution2D)       (None, 64, 64, 3)     12          lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 30, 30, 24)    1824        color_conv[0][0]                 
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 29, 29, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 13, 13, 36)    21636       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 5, 48)      43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 3, 64)      27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 1, 64)      36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 64)            0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          75660       flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================
Total params: 329,091
Trainable params: 329,091
Non-trainable params: 0

```

I have normalized the data to -0.5 to 0.5
I have added drop out after first fully connected layer
I have learning rate 0.0001, adam optimizer and mse as loss function
At the end of 30 epoch I have training loss of 0.02

#### Training data generation
```
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
```

Link to the [video](https://github.com/VenkatRepaka/CarND-Behavioral-Cloning-P3/blob/master/run.mp4)

### Future Work
The model I have employed could only work on simple track
I would fine tune the model so that it works on hard track.
My current video works on speed of 9. I would also like to adjust the speed and throttle on the basic track.
For the model to work on hard video it needs to adjust speed and throttle also.