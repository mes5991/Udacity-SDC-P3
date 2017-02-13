from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
import cv2, numpy as np
from image_processing import *
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from keras.layers import Input, ELU

def nvidia_model(inputShape):
    model = Sequential()
    model.add(BatchNormalization(input_shape=inputShape))

    model.add(Convolution2D(24, 5, 5, border_mode='same', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(48, 3, 3, border_mode='same', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    return model

X_train, y_train, X_test, y_test = getData("C:\\Users\\mes59\\Documents\\Udacity\\SDC-Term1-Projects\\project 3\\data\\udacityData3\\data\\")

print("Data Loaded")
print("Size of training set:", len(X_train))
print("Size of testing set:", len(X_test))
input("Press ENTER to continue")

model = nvidia_model((64,64,3))
model.compile('adam', 'mse')
model.load_weights("model.h5")
print("Model compiled!")

# input("Press ENTER to train")
history = model.fit_generator(generator_data(X_train, y_train,128), nb_epoch=1, verbose=1, samples_per_epoch=20000, validation_data=(generator_data(X_test, y_test, 256)),nb_val_samples=len(X_test))
with open('modelasdf.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('modelasdf.h5')
