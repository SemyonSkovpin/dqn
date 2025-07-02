import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import os

import pygame

# --- Parameters ---
env_name = 'LunarLander-v3'
gravity = -10
main_net_path = 'main_net.pth'
reload_interval = 10  # seconds between reloads of the policy

# --- Pygame setup for displaying return ---
pygame.init()
font = pygame.font.SysFont("Arial", 32)
screen = None

def show_return_on_screen(img, episode_return):
    global screen
    img = np.transpose(img, (1, 0, 2))  # gym returns (H, W, C), pygame wants (W, H, C)
    surf = pygame.surfarray.make_surface(img)
    if screen is None:
        screen = pygame.display.set_mode((img.shape[0], img.shape[1]))
    screen.blit(surf, (0, 0))
    text = font.render(f"Return: {episode_return:.1f}", True, (255, 255, 255))
    screen.blit(text, (10, 10))
    pygame.display.flip()

def load_policy():
    if os.path.exists(main_net_path):
        return torch.load(main_net_path)
    else:
        raise FileNotFoundError(f"{main_net_path} not found.")

def select_action(policy, obs):
    obs = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        q_values = policy(obs)
        action = torch.argmax(q_values).item()
    return action

def main():
    env = gym.make(env_name, gravity=gravity, render_mode='rgb_array')
    obs, _ = env.reset()
    last_reload = 0
    policy = load_policy()
    episode_return = 0

    while True:
        # Reload policy if needed
        if time.time() - last_reload > reload_interval:
            try:
                policy = load_policy()
                print("Reloaded policy.")
            except Exception as e:
                print("Could not reload policy:", e)
            last_reload = time.time()

        action = select_action(policy, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

        img = env.render()
        show_return_on_screen(img, episode_return)

        # Handle pygame events to allow window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                return

        if terminated or truncated:
            obs, _ = env.reset()
            episode_return = 0

        time.sleep(0.03)  # Control speed

if __name__ == "__main__":
    main()