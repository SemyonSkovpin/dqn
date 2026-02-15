import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import os

main_net = torch.load('main_net.pth')

env = gym.make('LunarLander-v3', render_mode='human')
obs, _ = env.reset()
episode_return = 0

while True:
    act = torch.argmax(main_net(torch.as_tensor(obs))).item()
    obs, reward, terminated, truncated, _ = env.step(act)
    episode_return += reward
    print(episode_return)
    if terminated or truncated:
        obs, _ = env.reset()
