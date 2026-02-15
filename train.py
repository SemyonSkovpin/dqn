import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import os


#== Parameters ================================================#
env_name = 'LunarLander-v3'
seed = 1
main_net_path = 'main_net.pth'

mini_batch_size = 128
buffer_size_limit = 10000
steps_until_main_net_update = 10
steps_until_target_net_update = 500
steps_to_train = 500000
steps_before_learning_starts = 10000

gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
exploration_fraction = 0.5 # Fraction of total timesteps it takes from epsilon_start to epsilon_end
lr=2.5e-4


#== Environment and seeds ================================================#
env = gym.make(env_name, render_mode='rgb_array')
env.reset(seed=seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


#== Main q-network and target q-network ================================================#
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n


def make_mlp():
    return nn.Sequential(
        nn.Linear(observation_size, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, action_size)
    )


if os.path.exists(main_net_path):
    main_net = torch.load(main_net_path)
else:
    main_net = make_mlp()
    torch.save(main_net, main_net_path)
target_net = make_mlp()


def update_target_net():
    target_net.load_state_dict(main_net.state_dict())


update_target_net() # They are same from the start

optimiser = optim.Adam(main_net.parameters(), lr=lr)


#== Implement expsilon-greedy policy ================================================#
def get_action(observation):
    '''
    observation: numpy array returned by env.step()
    returns: integer action
    '''
    possible_actions = [i for i in range(env.action_space.n)]
    if np.random.random() < epsilon:
        return np.random.choice(possible_actions)

    observation = torch.as_tensor(observation, dtype=torch.float32)
    with torch.no_grad():
        q_star_per_each_action = main_net(observation)
        action = torch.argmax(q_star_per_each_action).item()
    return action


#== Epsilon decay ================================================#
def get_current_epsilon(t):
    slope = (epsilon_end - epsilon_start) / steps_to_train * exploration_fraction
    epsilon = max(epsilon_end, epsilon_start +  slope * t)
    return epsilon


#== Replay buffer ================================================#
replay_buffer = deque([], maxlen=buffer_size_limit)

Timestep   = namedtuple('timestep',   ["state", "action", "reward", "next_state", "done"]) 
'''
Data types of Timestep:
state/next_state is numpy array; 
action is int; 
reward is float; 
done is bool;
'''

Mini_batch = namedtuple('mini_batch', ["states", "actions", "rewards", "next_states", "dones"])


def record_timestep(timestep):
    replay_buffer.append(timestep)


def sample_a_mini_batch():
    '''
    returns: named tuple with 5 1d tensors
    '''
    mini_batch = random.sample(replay_buffer, mini_batch_size)
    mini_batch = list(zip(*mini_batch)) # Transpose

    # Convert list of ndarrays to ndarray because Creating a tensor from a list of numpy.ndarrays is extremely slow. 
    states = np.array(mini_batch[0])
    next_states = np.array(mini_batch[3])
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(mini_batch[1], dtype=torch.int64)
    rewards = torch.tensor(mini_batch[2], dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(mini_batch[4], dtype = torch.bool)
    
    return Mini_batch(states, actions, rewards, next_states, dones)


#== Use mini_batch to get loss ================================================#
def compute_loss(mini_batch):
    # Compute targets 
    v_star_of_next_states = torch.max(target_net(mini_batch.next_states), dim=1)[0]
    v_star_of_next_states = v_star_of_next_states * (~mini_batch.dones) 
    y = mini_batch.rewards + gamma * v_star_of_next_states 

    # Compute main_net's predictions
    predictions = main_net(mini_batch.states).gather(1, mini_batch.actions.unsqueeze(1)).squeeze(1)

    # Loss 
    loss = nn.functional.mse_loss(predictions, y)

    return loss



#== Training loop ================================================#
observation, _ = env.reset()

t = -1
while True:
    t += 1
    timesteps_passed = t + 1
    epsilon = get_current_epsilon(t)
    
    action = get_action(observation)
    
    new_observation, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated 

    timestep = Timestep(observation, action, reward, new_observation, done)
    record_timestep(timestep)

    observation = new_observation

    log_reward_for_plotting(reward)


    if done:
        observation, _ = env.reset()
        update_plot()

    training_started = timesteps_passed >= steps_before_learning_starts
    time_to_update_main = training_started and timesteps_passed % steps_until_main_net_update == 0
    time_to_update_target = training_started and timesteps_passed % steps_until_target_net_update == 0
    
    # Do weights update if its time to
    if time_to_update_main:
        mini_batch = sample_a_mini_batch()
        loss = compute_loss(mini_batch)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        torch.save(main_net, main_net_path)

    # Update target net if its time to
    if time_to_update_target:
        update_target_net()

 

