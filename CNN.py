# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 01:20:17 2022

Project: CNN

@author: Dinesh
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

# Data Preprocessing

training_set_loc = "D:/dataset/training_set"
test_set_loc = "D:/dataset/test_set" 

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,       #to dive the pixel by 255
                                   shear_range = 0.2,
                                   zoom_range = 0.2,       
                                   horizontal_flip = True) #Flip of the image

#train_datagen will pre-process the images, image augmentation to discorage overfitting

training_set = train_datagen.flow_from_directory(training_set_loc,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(test_set_loc,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#Building the CNN

#Step 1 - Convolution
# Initialising the CNN
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

#filters = no.of feature detectors, classic archetecure has 32 in 1st and 32 in 2nd layers, can be diffrent depending on the archetecture.
#kernel_size = size of the feature detector 3x3 matrix.
#input_shape = image shape (64x64) and 3 for rgb(colored) if B&W it should be 1, i.e. [64,64,3].

#Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# pool_size = feature map pool size 2x2 matrix.
# strides = pixel selection slide operation, no.of pixel to be traversed.

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
#input_shape is only required for intial input augmentation.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
#higher unit value better results

#Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#only one unit as this is a binary classification
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#only one unit as this is a binary classification

#Part 3 - Training the CNN
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 30 )

#Part 4 - Making a single prediction
test_image_1 = "D:/dataset/single_prediction/cat_or_dog_1.jpg"
test_image_2 = "D:/dataset/single_prediction/cat_or_dog_2.jpg"
def imgPred (test_image_loc):
    test_image = image.load_img(test_image_loc, target_size = (64, 64))
    test_image = image.img_to_array(test_image) # converts PIL(Python Imaging Library) image in np array
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image/255) #scaling using /255
    training_set.class_indices
    if result[0][0] > 0.5:
        pred = "Dog"
    else:
        pred = "Cat"
    return pred

print('result for test_image_1: ', imgPred(test_image_1))

print('result for test_image_2: ', imgPred(test_image_2))


