import time
import random

import torch
from sb3_contrib import MaskablePPO

from snake_game_custom_wrapper_cnn import SnakeEnv
from network_structures import DVNNetwork
import numpy as np
import matplotlib.pyplot as plt
import torch as th

if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"trained_models_cnn/snake_s7_l4_grow_g985_160000000_steps"

NUM_EPISODE = 1000

DESIRED_STATE_NAME = 'len39_state_2024_08_13_10_55_44.obj'

RENDER = False
FRAME_DELAY = 0.05 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

if RENDER:
    env = SnakeEnv(seed=seed, length = "random", is_grow=True, limit_step=True, silent_mode=False)
else:
    env = SnakeEnv(seed=seed, length = "random", is_grow=True, limit_step=True, silent_mode=True)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0

estimated_value = 0

terminal_length_distrib = np.zeros(146)
for episode in range(NUM_EPISODE):
    obs, _ = env.load_state(DESIRED_STATE_NAME)
    episode_reward = 0
    done = False
    truncate = False
    
    num_step = 0
    info = None

    sum_step_reward = 0

    retry_limit = 9
    print(f"=================== Episode {episode + 1} ==================")
    while not (done or truncate):
        model.policy.set_training_mode(False)
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        prev_mask = env.get_action_mask()
        prev_direction = env.game.direction
        num_step += 1
        obs, reward, done, truncate, info = env.step(action)

        if done:
            if info["snake_size"] == env.game.grid_size:
                print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
            else:
                last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
                print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

        elif info["food_obtained"]:
            # print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            sum_step_reward = 0 

        else:
            sum_step_reward += reward
            
        episode_reward += (0.94**env.total_steps) * reward
    
    estimated_value += (episode_reward - estimated_value) / (episode + 1)

    episode_score = env.game.score
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score
    
    snake_size = info["snake_size"] + 1
    print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}")
    print("VALUE OF", DESIRED_STATE_NAME, ':', estimated_value)
    terminal_length_distrib[snake_size] += 1
    total_reward += episode_reward
    total_score += env.game.score
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"=================== Summary ==================")
print(f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODE}")
print(terminal_length_distrib)
plt.bar(range(3, 146), terminal_length_distrib[3:146])
plt.show()
