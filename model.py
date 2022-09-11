#!/bin/env python3
#
# Trains the model which drives the car
#

import csv
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers

# Seed
keras.utils.set_random_seed(1234)

#
# Load data
#

# Load csv file with steering angle, speed, camera file locations
csv_log_file_name = "../self-driving-car-sim/data/driving_log.csv"
center_images = []
steering_angles = []
print("Opening log file: ",csv_log_file_name)
with open(csv_log_file_name) as csv_log_file: 
    reader = csv.reader(csv_log_file)
    for i,line in enumerate(reader): 

        # lines are 
        # center_image_file, left_image_file, right_image_file,
        # steering angle, sth(accelerator?), sth(brake?), sth(speed?)
        center_image = cv2.imread(line[0])
        center_images.append(center_image)
        steering_angles.append(float(line[3]))

print("Creating dataset")
X_train = np.array(center_images)
y_train = np.array(steering_angles)

#
# Augment data
#
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1,random_state=1234)
# this aug gives OOO on GPU
# # do nothing for now and repoduce rsult
# print("X_train shape:",X_train.shape) #(8567, 160, 320, 3)
# print("y_train shape:",y_train.shape) #(8567,)
# idx = np.arange(0, X_train.shape[0])
# rand_idx = np.random.choice(idx, X_train.shape[0]//2, replace=False)
# X_aug = X_train[rand_idx]
# y_aug = y_train[rand_idx]
# for i in range(0,len(X_aug)):
#     X_aug[i] = np.fliplr(X_aug[i])
#     y_aug[i] = -1.*y_aug[i]

# print("X_aug shape",X_aug.shape)
# print("y_aug shape",y_aug.shape)

# X_train = np.concatenate((X_train,X_aug))
# y_train = np.concatenate((y_train,y_aug))

# print("concatenated X_train shape",X_train.shape)
# print("concatenated y_train shape",y_train.shape)

#
# Create model
#
print("Creating model")

# Adapt data normalization
#normalizer = layers.Normalization(axis=-1)
#normalizer.adapt(X_train)
# # test normalization 
# # X_train has shape (9519, 160, 320, 3)
# norm_data = normalizer(X_train[:100,:,:,:])
# print("var: ",np.var(norm_data))
# print("mean: ",np.mean(norm_data))

#
# Input
#
inputs = keras.Input(shape=(160,320,3))

#
# Preprocessing
#
# Crop the sky, 40px from the top
#X = layers.Cropping2D(cropping=((40,0),(0,0)))(inputs)
# try 60 from above
# try 20 from below
X = layers.Cropping2D(cropping=((60,20),(0,0)))(inputs)

# Normalize data
#X = normalizer(X)
X = layers.Rescaling(scale=2.0/255,offset = -1.)(X)

#
# Conv + Dense + .. Model
#
def LeNet(X):
    # Gives 8.4e-4 train loss & 0.0022 val loss. 
    # while train loss decreases over epocs, 
    # val loss pretty stable.
    X = layers.Conv2D(filters=6,kernel_size=(5,5),strides=(1,1)
                ,padding='same',activation='relu')(X)
    X = layers.AveragePooling2D(pool_size=(2,2),strides=(2,2))(X)

    X = layers.Conv2D(filters=16,kernel_size=(5,5),strides=(1,1)
                ,padding='valid',activation='relu')(X)
    X = layers.AveragePooling2D(pool_size=(2,2) ,strides=(2,2))(X)

    X = layers.Flatten()(X)
    X = layers.Dense(120)(X)
    X = layers.Dense(84)(X)
    X = layers.Dense(1)(X)
    return X

def LeNetDrop(X):
    # Add a dropout layer to help generalizing
    X = layers.Conv2D(filters=6,kernel_size=(5,5),strides=(1,1)
                ,padding='same',activation='relu')(X)
    X = layers.AveragePooling2D(pool_size=(2,2),strides=(2,2))(X)

    X = layers.Conv2D(filters=16,kernel_size=(5,5),strides=(1,1)
                ,padding='valid',activation='relu')(X)
    X = layers.AveragePooling2D(pool_size=(2,2) ,strides=(2,2))(X)

    X = layers.Flatten()(X)
    X = layers.Dropout(rate=0.2)(X)
    X = layers.Dense(120)(X)
    X = layers.Dense(84)(X)
    X = layers.Dense(1)(X)
    return X

def LeNetDrop2(X):
    # Add a dropout layer to help generalizing
    X = layers.Conv2D(filters=6,kernel_size=(5,5),strides=(1,1)
                ,padding='same',activation='relu')(X)
    X = layers.AveragePooling2D(pool_size=(2,2),strides=(2,2))(X)

    X = layers.Conv2D(filters=16,kernel_size=(5,5),strides=(1,1)
                ,padding='valid',activation='relu')(X)
    X = layers.AveragePooling2D(pool_size=(2,2) ,strides=(2,2))(X)

    X = layers.Flatten()(X)
    X = layers.Dropout(rate=0.4)(X)
    X = layers.Dense(120)(X)
    X = layers.Dropout(rate=0.2)(X)
    X = layers.Dense(84)(X)
    X = layers.Dense(1)(X)
    return X
def LeNetDrop2Max(X):
    # Add a dropout layer to help generalizing
    # Max pool instead of average
    X = layers.Conv2D(filters=6,kernel_size=(5,5),strides=(1,1)
                ,padding='same',activation='relu')(X)
    X = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(X)

    X = layers.Conv2D(filters=16,kernel_size=(5,5),strides=(1,1)
                ,padding='valid',activation='relu')(X)
    X = layers.MaxPooling2D(pool_size=(2,2) ,strides=(2,2))(X)

    X = layers.Flatten()(X)
    X = layers.Dropout(rate=0.4)(X)
    X = layers.Dense(120)(X)
    X = layers.Dropout(rate=0.2)(X)
    X = layers.Dense(84)(X)
    X = layers.Dense(1)(X)
    return X

def AlexNet500(X):
    # AlexNet but only 512,512 dense instead of 4096,4096
    X = layers.Conv2D(filters=96,kernel_size=(11,11),strides=(4,4)
                ,padding='same',activation='relu')(X)
    X = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)

    X = layers.Conv2D(filters=256,kernel_size=(5,5),strides=(1,1)
                ,padding='same',activation='relu')(X)
    X = layers.MaxPooling2D(pool_size=(3,3) ,strides=(2,2))(X)

    X = layers.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1)
                ,padding='same',activation='relu')(X)
    X = layers.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1)
                ,padding='same',activation='relu')(X)
    X = layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1)
                ,padding='same',activation='relu')(X)
    X = layers.MaxPooling2D(pool_size=(3,3) ,strides=(2,2))(X)

    X = layers.Flatten()(X)
    X = layers.Dense(512)(X)
    X = layers.Dropout(rate=0.5)(X)
    X = layers.Dense(512)(X)
    X = layers.Dropout(rate=0.5)(X)
    X = layers.Dense(1)(X)
    return X

def Nvidia(X):
    # https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
    # Image is 160,320, cropping (above) makes it 120,320
    X = layers.Conv2D(filters=12,kernel_size=(5,5),strides=(2,2)
                ,padding='valid',activation='relu')(X)
    X = layers.Conv2D(filters=24,kernel_size=(5,5),strides=(2,2)
                ,padding='valid',activation='relu')(X)
    X = layers.Conv2D(filters=36,kernel_size=(5,5),strides=(2,2)
                ,padding='valid',activation='relu')(X)
    X = layers.Conv2D(filters=48,kernel_size=(5,5),strides=(2,2)
                ,padding='valid',activation='relu')(X)
    X = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1)
                ,padding='valid',activation='relu')(X)
    X = layers.Flatten()(X) # Nvidia's size 1152 here, ours 1920
    X = layers.Dropout(rate=0.4)(X)
    X = layers.Dense(1164)(X)
    X = layers.Dropout(rate=0.2)(X)
    X = layers.Dense(100)(X)
    X = layers.Dense(50)(X)
    X = layers.Dense(10)(X)
    X = layers.Dense(1)(X)
    return X

def LeNetDrop2XConv(X):
    # Add a dropout layer to help generalizing
    # Add an extra conv layer
    X = layers.Conv2D(filters=6,kernel_size=(5,5),strides=(1,1)
                ,padding='same',activation='relu')(X)
    X = layers.AveragePooling2D(pool_size=(2,2),strides=(2,2))(X)
    X = layers.Conv2D(filters=16,kernel_size=(5,5),strides=(1,1)
                ,padding='valid',activation='relu')(X)
    X = layers.AveragePooling2D(pool_size=(2,2) ,strides=(2,2))(X)

    X = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1)
                ,padding='valid',activation='relu')(X)
    X = layers.AveragePooling2D(pool_size=(2,2) ,strides=(2,2))(X)
    X = layers.Conv2D(filters=48,kernel_size=(3,3),strides=(1,1)
                ,padding='valid',activation='relu')(X)
    # max pooling here would be super bad
    # with new crop: dropping it also bad
    X = layers.AveragePooling2D(pool_size=(2,2) ,strides=(2,2))(X) 
    X = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1)
                ,padding='valid',activation='relu')(X)

    X = layers.Flatten()(X)
    X = layers.Dropout(rate=0.4)(X)
    X = layers.Dense(120)(X)
    X = layers.Dropout(rate=0.2)(X)
    X = layers.Dense(84)(X)
    X = layers.Dropout(rate=0.2)(X)
    X = layers.Dense(1)(X)
    return X

def LeNetDrop2XConvPoorRes(input):
    # Add a dropout layer to help generalizing
    # Add an extra conv layer
    conv1 = layers.Conv2D(filters=8,kernel_size=(5,5),strides=(1,1)
                ,padding='same',activation='relu'
                ,name='conv1')(input)
    av1 = layers.AveragePooling2D(pool_size=(2,2),strides=(2,2)
                ,name='av1')(conv1) # 40h, 160w, 6c

    # Conv Block - half h,w double c
    conv2_1 = layers.Conv2D(filters=16,kernel_size=(5,5),strides=(1,1)
                ,padding='same',activation='relu'
                ,name='conv2_1')(av1)
    av2_1 = layers.AveragePooling2D(pool_size=(2,2) ,strides=(2,2)
                ,name='av2_1')(conv2_1)
    conv2_2 = layers.Conv2D(filters=16,kernel_size=(5,5),strides=(1,1)
                ,padding='same'
                ,name='conv2_2')(av2_1) # no activation
    skip2 = layers.Conv2D(filters=16,kernel_size=(1,1),strides=(2,2)
                ,padding='same'
                ,name='skip2')(av1) # no activation
    add2 = layers.add([conv2_2,skip2]
                ,name='add2')
    act2 = layers.Activation(keras.activations.relu,
                name='act2')(add2)

    # Ident Block - h,w,c stays
    conv3_1 = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1)
                ,padding='same',activation='relu'
                ,name='conv3_1')(act2)
    conv3_2 = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1)
                ,padding='same'
                ,name='conv3_2')(conv3_1) # no activation
    skip3 = layers.Conv2D(filters=16,kernel_size=(1,1),strides=(1,1)
                ,padding='same'
                ,name='skip3')(act2) # no activation
    add3 = layers.add([conv3_2,skip3]
                ,name='add3')
    act3 = layers.Activation(keras.activations.relu,
                name='act3')(add3)

    # COnv block
    conv4_1 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1)
                ,padding='same',activation='relu'
                ,name='conv4_1')(act3)
    av4_1 = layers.AveragePooling2D(pool_size=(2,2) ,strides=(2,2)
                ,name='av4_1')(conv4_1)
    conv4_2 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1)
                ,padding='same'
                ,name='conv4_2')(av4_1) # no activation
    skip4 = layers.Conv2D(filters=32,kernel_size=(1,1),strides=(2,2)
                ,padding='same'
                ,name='skip4')(act3) # no activation
    add4 = layers.add([conv4_2,skip4]
                ,name='add4')
    act4 = layers.Activation(keras.activations.relu,
                name='act4')(add4)

    # ident block
    conv5_1 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1)
                ,padding='same',activation='relu'
                ,name='conv5_1')(act4)
    conv5_2 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1)
                ,padding='same'
                ,name='conv5_2')(conv5_1) # no activation
    skip5 = layers.Conv2D(filters=32,kernel_size=(1,1),strides=(1,1)
                ,padding='same'
                ,name='skip5')(act4) # no activation
    add5 = layers.add([conv5_2,skip5]
                ,name='add5')
    act5 = layers.Activation(keras.activations.relu,
                name='act5')(add5)

    # conv block
    conv6_1 = layers.Conv2D(filters=48,kernel_size=(3,3),strides=(1,1)
                ,padding='same',activation='relu'
                ,name='conv6_1')(act5)
    av6_1 = layers.AveragePooling2D(pool_size=(2,2) ,strides=(2,2)
                ,name='av6_1')(conv6_1)
    conv6_2 = layers.Conv2D(filters=48,kernel_size=(3,3),strides=(1,1)
                ,padding='same'
                ,name='conv6_2')(av6_1) # no activation
    skip6 = layers.Conv2D(filters=48,kernel_size=(1,1),strides=(2,2)
                ,padding='same'
                ,name='skip6')(act5) # no activation
    add6 = layers.add([conv6_2,skip6]
                ,name='add6')
    act6 = layers.Activation(keras.activations.relu,
                name='act6')(add6)

    conv7 = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1)
                ,padding='valid',activation='relu'
                ,name='conv7')(act6)
    av7 = layers.AveragePooling2D(pool_size=(2,2) ,strides=(2,2)
                ,name='av7')(conv7)

    X = layers.Flatten()(av7)
    X = layers.Dropout(rate=0.4)(X)
    X = layers.Dense(120)(X)
    X = layers.Dropout(rate=0.2)(X)
    X = layers.Dense(84)(X)
    X = layers.Dense(1)(X)
    return X


outputs = LeNetDrop2XConv(X)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

#
# Compile and train
#
#initial_learning_rate = 0.001 # 0.001 default for Adam
initial_learning_rate = 0.1 # 0.001 default for Adam
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.5,
    staircase=True)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='mse',optimizer='adam')

print("Training model")
# model.fit(X_train,y_train,validation_data=(X_test,y_test),shuffle=True,
#             epochs=10,batch_size=4)
model.fit(X_train,y_train,validation_data=(X_test,y_test),shuffle=True,
            epochs=10,batch_size=4)

print("Saving model")
model.save('model.h5')

