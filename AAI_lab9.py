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

GAMA = 0.95
ACTION_SPACE_SIZE = 4
LEARNING_RATE = 0.02
EPOCHS = 10
EPOCH_SIZE = 100
BACH_SIZE = 32
OBSERVATION = 1000
EPSILON_UP = 0.1
EPSILON_LOW = 0.01
MEMORY_SIZE = 1000000
EXPLORATION_SIZE = 3000000


def setup():
    global env
    env = gym.make('Breakout-v0')
    print('action space: ', env.action_space)
    print('observation space', env.observation_space)


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


'''observe is the first term 'observation' given by env.step()
we resize the input to an 84*84 size image including to the paper
at last we wan to get a 4 frame data which contain the imformation of motion'''
def input_process(observe, state, init=False):
    input = skimage.color.rgb2grey(observe)
    input = skimage.transform.resize(input, (110,84), mode='constant', preserve_range=True)
    input = skimage.util.crop(input, ((19,7),(0,0)))

    if init==True:
        state = np.stack((input, input, input, input), axis=-1)
        state = state.reshape(1,state.shape[0],state.shape[1],state.shape[2])
    else:
        input = input.reshape(1,input.shape[0],input.shape[1],1)
        np.append(input, state[:,:,:,:3], axis=3)
    return state

def epoch_initialize(env):
    env.reset()
    observation, reward, done, info = env.step(0)

    state = input_process(observation, None, True)
    epsilon = EPSILON_UP

    return state, epsilon

'''In this function, we use minibatch coming from dataset 
to calculate the loss for the cov network to learn. 
X is the input image(state)
Y is the value list of Q as the effect of each action
There is something important that in DQL, the loss function 
is always changing because Q value will update at every episod'''
def model_train(model, Dataset, loss):
    minibatch = np.random.sample(Dataset, BACH_SIZE)
    for i in range(BACH_SIZE):
        state_prior, action, reward, state_post, done = minibatch[i]
        if 'X' not in dir():
            X = np.zeros((BACH_SIZE,state_prior[0],state_prior[1],state_prior[2]))
        if 'Y' not in dir():
            Y = np.zeros((BACH_SIZE,ACTION_SPACE_SIZE))
        X[i] = state_prior
        Y[i] = model.predict(state_prior)
        Q_post = model.predict(state_post)

        if done:
            Y[i, action] = reward
        else: Y[i, action] = reward + GAMA*np.max(Q_post)
    loss += model.train_on_batch(X, Y)
    return loss



# we train the model EPOCHS times. In each epoch, we do traing by episod steps
def model_run(model, env):
    global epsilon
    for epocch in range(EPOCHS):
        reward_total = 0
        episod = 0
        Dataset = deque()
        state_prior, epsilon = epoch_initialize(env)
        for t in range(EPOCH_SIZE):
            loss = 0
            action = 0
            reward = 0

            if random.random()<=epsilon:
                action = env.action_space.sample()
            else:
                Q_list = model.predict(state_prior)
                action = np.argmax(Q_list)
            if epsilon>EPSILON_LOW:
                epsilon -= (EPSILON_UP-EPSILON_LOW)/EXPLORATION_SIZE

            observation, reward, done, info = env.step(action)
            if done:
                episod +=1
                print('game '+episod+' is over')
                break
            state_post = input_process(observation, state_prior)
            reward_total += reward
            Dataset.append((state_prior, action, reward, state_post, done))
            if len(Dataset)>MEMORY_SIZE:
                Dataset.popleft()
            if t==OBSERVATION:
                print('--Observation is over, training now')
            if t>OBSERVATION:
                loss = model_train(model, Dataset, loss)
            state_prior = state_post



if __name__ == '__main__':
    #config = tf.ConfigProto
    #sess = tf.Session(config=config)
    from keras import backend as K
    #K.set_session(sess)
    #K.set_image_dim_ordering('tf')
    model = model()
    setup()
    model_run(model, env)
