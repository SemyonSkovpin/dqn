import torch
import gymnasium as gym
import pygame
import numpy as np

# === Define model class ===
class DQN(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)

# === Initialize environment ===
gravity = -10
env = gym.make("LunarLander-v3", gravity=gravity, render_mode="rgb_array")
obs, _ = env.reset()  # Reset before rendering
frame = env.render()
window_size = frame.shape[:2][::-1]

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

# === Load entire model (CPU-only) ===
model = torch.load("playing_assistant.pth", map_location="cpu")
model.eval()

# === Initialize pygame ===
pygame.init()
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Lunar Lander: Human Override + AI")
font = pygame.font.SysFont("Arial", 24)

# === Key mappings (arrows + WASD) ===
key_action_map = {
    pygame.K_LEFT: 1, pygame.K_a: 1,
    pygame.K_RIGHT: 3, pygame.K_d: 3,
    pygame.K_UP: 2, pygame.K_w: 2,
    pygame.K_DOWN: 0
}

clock = pygame.time.Clock()

# === Track held keys for continuous human control ===
held_keys = set()

# === Infinite episode loop ===
while True:
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        human_action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                exit()
            elif event.type == pygame.KEYDOWN and event.key in key_action_map:
                held_keys.add(event.key)
            elif event.type == pygame.KEYUP and event.key in key_action_map:
                held_keys.discard(event.key)

        # If any mapped key is held, use the first one found
        for key in held_keys:
            if key in key_action_map:
                human_action = key_action_map[key]
                break

        # Use model if no human input
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(obs_tensor)
        model_action = int(torch.argmax(q_values).item())
        action = human_action if human_action is not None else model_action

        # Take environment step
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render and draw reward overlay
        frame = env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        text = font.render(f"Total Reward: {int(total_reward)}", True, (255, 255, 255))
        screen.blit(text, (10, 10))

        # === Draw overlay for currently pressed player actions ===
        action_names = {1: "LEFT", 2: "UP", 3: "RIGHT"}
        pressed = set()
        for key in held_keys:
            if key in key_action_map:
                act = key_action_map[key]
                if act in action_names:
                    pressed.add(action_names[act])
        if pressed:
            pressed_str = " + ".join(sorted(pressed))
            overlay = font.render(f"Player: {pressed_str}", True, (255, 200, 0))
            screen.blit(overlay, (10, 40))

        pygame.display.flip()

        clock.tick(15)  # ~15 FPS for human pace
