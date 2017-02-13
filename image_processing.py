import cv2
import numpy as np
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random

def augment_brightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    augment = .25+np.random.uniform(low=0.0, high=1.0)
    img[:,:,2] = img[:,:,2]*augment
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def translate_image(image,steer,trans_range):
    i,j = image.shape[0],image.shape[1]
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(j,i))

    return image_tr,steer_ang

def flip_image(img):
    return cv2.flip(img, 1)

def reshape_image(img):
    shape = img.shape
    img = img[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return img

def image_augmentation_flow(img, angle):
    img, angle = translate_image(img, angle, 100)
    img = augment_brightness(img)
    img = reshape_image(img)
    img = np.array(img)
    if np.random.randint(2):
        img = flip_image(img)
        angle = -angle
    return img, angle


def getData(data_path):
    csv_path = [data_path + "driving_log.csv", data_path + "driving_log_gasoto.csv"]
    steer_offset = .25
    X_train = []
    y_train = []

    for i in range(len(csv_path)):
        with open(csv_path[i], 'r') as f:
            reader = csv.reader(f)
            row = next(reader) #move past the header row
            for row in reader: #format: center name, left name, right name, steering, throttle, brake, speed
                # if abs(float(row[3])) < .1:
                #     continue
                # if float(row[3]) == 0:
                #     if random.randint(0,3) != 0:
                #         continue
                row[0] = row[0].strip()
                row[1] = row[1].strip()
                row[2] = row[2].strip()
                X_train.append(row[0])
                y_train.append(float(row[3]))
                X_train.append(row[1])
                y_train.append(float(row[3]) + steer_offset)
                X_train.append(row[2])
                y_train.append(float(row[3]) - steer_offset)

    X_train, y_train = shuffle(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.20)
    return(X_train, y_train, X_test, y_test)

def generator_data(data, angle, batch_size):
    index = np.arange(len(data))
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size), dtype = np.float32)
    while 1:
        for i in range(batch_size):
            random = int(np.random.choice(index,1))
            img = cv2.imread(data[random])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            batch_train[i], batch_angle[i] = image_augmentation_flow(img, angle[random])
        yield (batch_train, batch_angle)

def display_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
