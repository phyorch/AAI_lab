import numpy as np
import keras
from keras.models import Sequential
from keras.layers import *
from keras.initializers import *
from keras.optimizers import *
import tensorflow as tf
import skimage
from skimage import color
from skimage import transform
from skimage import util
from skimage import exposure
from skimage.viewer import ImageViewer
import gym
import random
from collections import deque
import json
import argparse
import time

# INTERNAL
#import lib.notify as notify
#import lib.stats as stats
#from lib.hyperparams import *




ACTION_SPACE_SIZE = 4
LEARNING_RATE = 0.2

def model():
    'build our model'
    model = Sequential()
    model.add(Conv2D(32, (8,8), strides=(4,4), input_shape=(84,84,4), padding='same', name='conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', name='conv2'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv3'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, name='fc1'))
    model.add(Activation('relu'))
    model.add(Dense(ACTION_SPACE_SIZE, name='fc2'))
    adam = Adam(lr= LEARNING_RATE)
    model.compile(loss='mse', optimizer= adam)
    return model
