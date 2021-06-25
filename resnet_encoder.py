#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:40:09 2021

@author: SAchit
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist
from keras.models import Model
# import os
# import random
import numpy as np
# tf.keras.optimizers.Adadelta(
#     learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta", **kwargs
# )

(x_train,y_train),(x_test,y_test) = mnist.load_data()
seed = 42
np.random.seed = seed

IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) 
#s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
lr1 = tf.keras.layers.LeakyReLU(alpha=0.3)(p1)
bn1 = tf.keras.layers.BatchNormalization()(lr1) #SHORTCUT


'''THIS IS THE REPEATING LAYER'''
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(bn1)
lr2 = tf.keras.layers.LeakyReLU(alpha=0.3)(c2)
bn2 = tf.keras.layers.BatchNormalization()(lr2)

c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn2)
lr3 = tf.keras.layers.LeakyReLU(alpha=0.3)(c2)
bn3 = tf.keras.layers.BatchNormalization()(lr2)



add1 = tf.keras.layers.Add()([bn1, bn3])
bn4 = tf.keras.layers.BatchNormalization()(add1)

c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn4)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
lr4 = tf.keras.layers.LeakyReLU(alpha=0.3)(p4)
bn4 = tf.keras.layers.BatchNormalization()(lr4)#SHORTCUT


'''THIS IS THE REPEATING LAYER'''
c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(bn4)
lr5 = tf.keras.layers.LeakyReLU(alpha=0.3)(c5)
bn5 = tf.keras.layers.BatchNormalization()(lr5)

c6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(bn5)
lr6 = tf.keras.layers.LeakyReLU(alpha=0.3)(c6)
bn6 = tf.keras.layers.BatchNormalization()(lr6)



add2 = tf.keras.layers.Add()([bn4, bn6])
bn6 = tf.keras.layers.BatchNormalization()(add2)

x = tf.keras.layers.Flatten()(bn6)
x = tf.keras.layers.Dense(3136, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)

prediction = tf.keras.layers.Dense(10,activation=tf.nn.softmax)(x)
model = Model(inputs=inputs, outputs=prediction)


model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta"),
  metrics=['accuracy'])

results = model.fit(x_train, y_train, validation_split=0.1, epochs=25)