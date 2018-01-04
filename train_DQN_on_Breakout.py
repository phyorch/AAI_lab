"""Train a Deep Q-Network (i.e., DQN) Model on the Breakout Game
From the OpenAI Gym Reinforcement Learning Environment.

----------
Reference:
----------
1. https://github.com/openai/gym
2. https://gym.openai.com/envs/Breakout-v0/
3. https://keras.io/
4. https://keras.io/getting-started/sequential-model-guide/
5. Mnih V, Kavukcuoglu K, Silver D, et al. 
    Human-level control through deep reinforcement learning. 
    Nature, 2015, 518(7540): 529-533.
"""


import os
import time
import random
from collections import deque # FIFO

import numpy as np
from skimage import color
from skimage import transform

import gym

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop


class DQN:  # Deep Q-learning Neural Network
    def __init__(self, batch_size, memory_size, state_dim, num_actions):
        self.batch_size = batch_size
        self.memory_size = memory_size  # replay memory size
        self.state_dim = state_dim   # state dimension
        self.num_actions = num_actions  # number of actions
        self.experience = deque(maxlen=memory_size)  # cache for further [online] batch train
        self.dnn_model = self._design_dnn()
        self.target_dnn_model = self._design_target_dnn()
        self.x_train = np.zeros((batch_size, state_dim[0], state_dim[1], state_dim[2]))
        self.next_state = np.zeros((batch_size, state_dim[0], state_dim[1], state_dim[2]))
    
    def _design_dnn(self):
        dnn_model = Sequential()
        dnn_model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                             input_shape=self.state_dim))
        dnn_model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        dnn_model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        dnn_model.add(Flatten())
        dnn_model.add(Dense(512, activation='relu'))
        dnn_model.add(Dense(num_actions))
        dnn_model.compile(optimizer=RMSprop(lr=0.00025, epsilon=0.01), loss='mse')
        return dnn_model
    
    def _design_target_dnn(self):
        target_dnn_model = self._design_dnn()
        target_dnn_model.set_weights(self.dnn_model.get_weights())
        return target_dnn_model
    
    def act(self, curr_state, action_space, explore_ratio):
        curr_state = np.float32(curr_state / 255.0)
        if np.random.rand() < explore_ratio: # randomly explore some new action
            return action_space.sample()
        else:  # return the action with the maximal Q-value
            return np.argmax(self.dnn_model.predict(curr_state)[0]) 
    
    def cache(self, curr_state, curr_action, next_reward, next_state, done):
        self.experience.append((curr_state, curr_action, next_reward, next_state, done))
    
    def train(self, game_over_reward=0.0, discount_factor=0.99):
        batch = random.sample(self.experience, self.batch_size)
        for batch_ind in range(self.batch_size):
            self.x_train[batch_ind] =  np.float32(batch[batch_ind][0] / 255.0) # curr_state
            self.next_state[batch_ind] = np.float32(batch[batch_ind][3] / 255.0) # next_state
        # update Q-value
        y_train = self.dnn_model.predict(self.x_train)
        next_state_q_value = self.target_dnn_model.predict(self.next_state)
        for batch_ind in range(self.batch_size):
            curr_action, next_reward, done = \
                int(batch[batch_ind][1]), batch[batch_ind][2], batch[batch_ind][4]
            if done:
                y_train[batch_ind, curr_action] = game_over_reward
            else:
                y_train[batch_ind, curr_action] = next_reward + \
                    discount_factor * np.amax(next_state_q_value[batch_ind])
        self.dnn_model.fit(self.x_train, y_train, batch_size=self.batch_size, verbose=0)


def _process_state(curr_stack, state, state_dim, game_begin_flag=False, num_stacks=4):
    # convert RGB to Gray
    state = color.rgb2gray(state)
    # rescale the image frame + convert the data type for efficient memory
    state = np.uint8(transform.resize(state[25:, :], (84, 84), mode='constant') * 255)
    # create a stack for the most recent image frames
    if game_begin_flag == True:
        for _ in range(num_stacks):
            curr_stack.append(state)
    else:
        curr_stack.append(state)
    state = np.stack(curr_stack, axis=2)
    # add one dimension for inputs of DQN
    return np.reshape(state, [1, state_dim[0], state_dim[1], state_dim[2]])


def _print_env(env_name, env, num_actions, state_dim):
    # print some basic environment information
    print('env =', env_name)
    print('action space =', env.action_space)
    print('num_actions = ', num_actions)
    print('observation space =', env.observation_space)
    print('state_dim = ', state_dim)    


def _set_params(): # set *trial + environment + model* parameters
    args = {} # dict for setting params
    # params for trials
    args['train_flag'] = False # flag to denote to train DQN (True or False) *
    args['env_name'] = 'BreakoutDeterministic-v4'
    args['num_episodes'] = 1 # number of episodes
    args['num_frame_in_stack'] = 4 # agent history length
    # params for DNN
    args['update_target_dnn_freq'] = 10000 # frequency of updating the target DNN model
    args['batch_size'] = 32 # minibatch size
    args['memory_size'] = 400000 # replay memory size
    args['load_weights'] = True # flag to load weights
    args['weights_filepath'] = './dnn_weights_for_breakout_v0.h5' # filepath for weights
    if args['train_flag']: # only when train
        args['num_start_train'] = 50000 # number of starting train
        args['save_weights'] = True # save weights for DNN
        args['save_weights_freq'] = 1000 # frequency for saving weights
        args['render_env_flag'] = False # flag to visualize the environment
        args['max_no_actions'] = 30 # maximum of no actions at the early stage of each episode
        args['max_explore'] = 1000000 # maximum of exploration
        args['explore_ratio'] = 1.0 # explore_ratio
        args['init_explore_ratio'] = 1.0 # initial exploration ratio
        args['final_explore_ratio'] = 0.1 # final exploration ratio
        args['explore_decay_ratio'] = \
            (args['init_explore_ratio'] - args['final_explore_ratio']) / args['max_explore']
    else: # only when evaluation
        args['save_weights'] = False
        args['render_env_flag'] = True
        args['evaluate_explore_ratio'] = 0.01  # exploration ratio (or 0.01)
    # params for logging and dubugging
    args['freq_print_episodes_log'] = 1 # frequency of printing episodes logging
    return args


if __name__ == '__main__':
    total_run_time_start = time.time() # timing
    args = _set_params()
    print('args =', args)
    
    # build a Gym environment
    env = gym.make(args['env_name'])
    ## note that a new state dimension instead of the original one is used
    num_actions, state_dim = env.action_space.n, (84, 84, 4)
    _print_env(args['env_name'], env, num_actions, state_dim)
    
    # build a QDN-based agent
    dqn_agent = DQN(args['batch_size'], args['memory_size'], state_dim, num_actions)
    print(dqn_agent.dnn_model.summary())
    print(dqn_agent.target_dnn_model.summary())
    if args['load_weights'] == True and os.path.isfile(args['weights_filepath']):
        dqn_agent.dnn_model.load_weights(args['weights_filepath'])
    
    # [online] batch train the agent
    curr_stack = deque(maxlen=args['num_frame_in_stack']) # cache recent frames
    step_sum = 0 # step sum for all episodes
    for e in range(args['num_episodes']):
        run_time_start = time.time() # timing for each episode
        episode_step_sum = 0 # step sum of each episode
        episode_max_q_values = 0.0 # maximum of Q-value of each episode
        episode_reward_sum = 0.0 # summary rewards of each episode
        # begin the game
        curr_state, done = env.reset(), False
        if args['train_flag']: # only when train
            ## aim: create different initial random conditions
            ##      to avoid some local optimum at the early stage
            for _ in range(random.randint(1, args['max_no_actions'])):
                curr_state, _, _, _ = env.step(1) # 1 -> just do nothing
        # process state (i.e., images) for faster computation speed
        curr_state = _process_state(curr_stack, curr_state, state_dim,
                                   game_begin_flag=True, num_stacks=args['num_frame_in_stack'])
        while not done:
            if args['render_env_flag']:
                env.render()  # visualize the environment
            # act based on the current state
            if args['train_flag']: # only when train
                if step_sum < args['max_explore']: # for the early stage
                    args['explore_ratio'] -= args['explore_decay_ratio']
                else: # for the later stage
                    args['explore_ratio'] = args['final_explore_ratio']
            else: # only when evaluation
                args['explore_ratio'] = args['evaluate_explore_ratio']
            curr_action = dqn_agent.act(curr_state, env.action_space, args['explore_ratio'])
            step_sum, episode_step_sum = step_sum + 1, episode_step_sum + 1
            # get the next state
            next_state, next_reward, done, _ = env.step(curr_action)
            episode_reward_sum += next_reward
            next_state = _process_state(curr_stack, next_state, state_dim, 
                                       game_begin_flag=False, num_stacks=args['num_frame_in_stack'])
            episode_max_q_values += \
                np.amax(dqn_agent.dnn_model.predict(np.float32(curr_state / 255.0))[0])
            # cache experience
            dqn_agent.cache(curr_state, curr_action, next_reward, next_state, done)
            curr_state = next_state
            if args['train_flag'] and len(dqn_agent.experience) >= args['num_start_train']:
                dqn_agent.train()
                if step_sum % args['update_target_dnn_freq'] == 0:
                    dqn_agent.target_dnn_model.set_weights(dqn_agent.dnn_model.get_weights())
                if args['save_weights'] and ((e > 0) and (e % args['save_weights_freq'] == 0)):
                    dqn_agent.dnn_model.save_weights(args['weights_filepath'], overwrite=True)
            if done: # game over
                episode_run_time = time.time() - run_time_start
                if e % args['freq_print_episodes_log'] == 0:
                    print('> episode %07d - steps %07d : reward_sum %07d + run_time %09.2e + avg_Q_value %09.2e' \
                          % (e, episode_step_sum, episode_reward_sum, episode_run_time, 
                             episode_max_q_values / episode_step_sum))
                    break
    total_run_time_end = time.time()
    print('* run time * : {:09.2f} + * total steps * : {:09d}'.format(
        total_run_time_end - total_run_time_start, step_sum))
