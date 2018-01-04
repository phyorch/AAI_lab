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

import matplotlib as plt
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
EPOCH_SIZE = 30000
REPORT = 200
BACH_SIZE = 32
OBSERVATION = 1000
EPSILON_UP = 0.2
EPSILON_LOW = 0.01
MEMORY_SIZE = 1000000
EXPLORATION_SIZE = 3000000
FILE_PATH = '/home/phyorch/Learning/AAI_lab/model.h5'
WEIGHTS_PATH = '/home/phyorch/Learning/AAI_lab/model.h5'


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
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.imshow(input)
    # plt.show()
    if init==True:
        state = np.stack((input, input, input, input), axis=-1)
        state = state.reshape(1,state.shape[0],state.shape[1],state.shape[2])
    else:
        input = input.reshape(1,input.shape[0],input.shape[1],1)
        np.append(input, state[:,:,:,:3], axis=3)
    return state

def epoch_initialize(env):
    env.reset()
    observation, reward, done, info = env.step(env.action_space.sample())

    state = input_process(observation, None, True)
    #epsilon = EPSILON_UP

    return state, reward

'''In this function, we use minibatch coming from dataset 
to calculate the loss for the cov network to learn. 
X is the input image(state)
Y is the value list of Q as the effect of each action
There is something important that in DQL, the loss function 
is always changing because Q value will update at every episod'''
def model_train(model, Dataset):
    minibatch = random.sample(Dataset, BACH_SIZE)
    for i in range(BACH_SIZE):
        state_prior, action, reward, state_post, done = minibatch[i]
        if 'X' not in dir():
            X = np.zeros((tuple([BACH_SIZE])+ state_prior[0].shape))
        if 'Y' not in dir():
            Y = np.zeros((BACH_SIZE,ACTION_SPACE_SIZE))
        X[i] = state_prior
        #Y[i] = model.predict(state_prior)
        Q_post = model.predict(state_post)

        if done:
            Y[i, action] = reward
        else: Y[i, action] = reward + GAMA*np.max(Q_post)
    model.fit(X, Y, BACH_SIZE, verbose=0)
    #loss += model.train_on_batch(X, Y)
    #return loss



'''we train the model EPOCHS times. In each epoch, we do traing by episod steps'''
def model_run(model, env):
    #report = 0
    global epsilon
    epsilon = EPSILON_UP
    for epocch in range(EPOCHS):
        print('---------new epochs---------')
        #game_reward = 0
        episod = 0
        Dataset = deque()
        state_prior, game_reward = epoch_initialize(env)
        for t in range(EPOCH_SIZE):
            env.render()
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
            if reward!=0:
                print(reward)
            if done:
                print('runing times:  ', time.time())
                print('total reward:  ', game_reward)
                print('current epsilon:  ', epsilon)
                print('game ', episod, ' is over')
                state_prior, game_reward = epoch_initialize(env)
                model.save_weights(FILE_PATH)
                continue
            state_post = input_process(observation, state_prior)
            game_reward += reward
            Dataset.append((state_prior, action, reward, state_post, done))
            if len(Dataset)>MEMORY_SIZE:
                Dataset.popleft()
            if t==OBSERVATION:
                print('--Observation is over, training now')
            if t>OBSERVATION:
                model_train(model, Dataset)
            '''if t/REPORT==0:
                print('runing times:  ', time.time())
                print('total reward:  ',game_reward)
                print('current epsilon:  ', epsilon)'''
            state_prior = state_post
            if epocch==EPOCHS-1:
                model.save_weights(FILE_PATH)



if __name__ == '__main__':
    total_run_time_start = time.time()
    setup()
    # load weight
    #config = tf.ConfigProto()
    #sess = tf.Session(config=config)
    #from keras import backend as K
    #K.set_session(sess)
    #K.set_image_dim_ordering('tf')
    model = model()
    model.load_weights(WEIGHTS_PATH)
    model_run(model, env)

