import time
import random

import torch
from sb3_contrib import MaskablePPO

from snake_game_custom_wrapper_cnn import SnakeEnv
from network_structures import DVNNetwork
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import os

if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"coaches/snake_ob_len3_max130.zip"

NUM_EPISODE = 300

RENDER = True
FRAME_DELAY = 0.01 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5
VALUE_MODEL_NAMES = [
    # {"actor": "trained_models_cnn/snake21_s1_len3.zip", "critic": "trained_models_value/DVN_len3_to_BOSS_final.zip"},
    # {"actor": "trained_models_cnn/snake21_len80_max160_44000000_steps.zip", "critic": "trained_models_value/DVN_len80toBOSS_final.zip"},
    # {"actor": "trained_models_cnn/snake21_len300_please_success_38000000_steps.zip", "critic": "trained_models_value/DVN_len300_to_BOSS_final.zip"},
    # {"actor": "trained_models_cnn/snake21_len350loads_please_success_124000000_steps.zip", "critic": "trained_models_value/DVN_load_to_BOSS_final.zip"},
    # {"actor": "random_feature_extractor.zip", "critic": "trained_models_value/DVN_len80toBOSS_please_success_final.zip"},
]
AC_MODEL_NAMES = [
    # "trained_models_cnn/mc_value_evaluation_len3_in_len70max160_9000000_steps"
]

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

# Specify the directory
directory = "./game_states"

# Get the list of filenames in the specified directory
state_name_list = [filename for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename))]
state_name_list = [
            "len366_state_2024_08_15_08_48_32.obj",
            "len369_state_2024_08_15_08_49_35.obj"
        ]

if RENDER:
    env = SnakeEnv(seed=seed, length = 120, is_grow=False, limit_step=True, silent_mode=False)
else:
    env = SnakeEnv(seed=seed, length = 120, is_grow=False, limit_step=True, silent_mode=True)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)
if VALUE_MODEL_NAMES != []:
    value_models = []
    for vmn in VALUE_MODEL_NAMES:
        value_model = DVNNetwork(old_model_name=vmn["actor"]).to('cuda')
        value_model.load_state_dict(th.load(vmn["critic"]))
        value_model.eval()
        value_models.append(value_model)
if AC_MODEL_NAMES != []:
    ac_models = []
    for acn in AC_MODEL_NAMES:
        ac_model = MaskablePPO.load(acn)
        ac_model.policy.set_training_mode(False)
        ac_models.append(ac_model)

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0
terminal_length_distrib = np.zeros(443)
for episode in range(NUM_EPISODE):
    obs, _ = env.reset(9487, None)
    episode_reward = 0
    done = False
    truncate = False
    
    num_step = 0
    info = None

    sum_step_reward = 0

    retry_limit = 9
    print(f"=================== Episode {episode + 1} ==================")
    while not (done or truncate):
        # if info != None and info["snake_size"] >= 390:
        #     is_save = input()
        #     if is_save == 's':
        #         env.save_state()

        model.policy.set_training_mode(False)
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        if VALUE_MODEL_NAMES != [] or AC_MODEL_NAMES != []:
            obs, _ = model.policy.obs_to_tensor(obs)
        if VALUE_MODEL_NAMES != []:
            # Initialize a variable to track the maximum value and its index
            max_value = -np.inf
            max_index = -1
            values = []
            # First, identify the maximum value and its index
            for i, vmn in enumerate(VALUE_MODEL_NAMES):
                current_value = value_models[i](obs).item()
                values.append(current_value)
                if current_value > max_value:
                    max_value = current_value
                    max_index = i
            # Then, print the values, marking the one with the maximum value in red
            for i, vmn in enumerate(VALUE_MODEL_NAMES):
                value = values[i]
                if i == max_index:
                    # Mark the maximum value in red
                    print(f"\033[91m{vmn['actor']}'s policy value by TD: {value}\033[0m")
                else:
                    print(f"{vmn['actor']}'s policy value by TD: {value}")

        if AC_MODEL_NAMES != []:
            # Initialize a variable to track the maximum value and its index
            max_value = -np.inf
            max_index = -1
            values = []
            # First, identify the maximum value and its index
            for i in range(len(AC_MODEL_NAMES)):
                current_value = ac_models[i].policy.predict_values(obs).item()
                values.append(current_value)
                if current_value > max_value:
                    max_value = current_value
                    max_index = i
            # Then, print the values, marking the one with the maximum value in red
            for i, acn in enumerate(AC_MODEL_NAMES):
                value = values[i]
                if i == max_index:
                    # Mark the maximum value in red
                    print(f"\033[91m{acn}'s policy value by TD: {value}\033[0m")
                else:
                    print(f"{acn}'s policy value by TD: {value}")
        if VALUE_MODEL_NAMES != [] or AC_MODEL_NAMES != []:
            print("========================================================")
            input()
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
            print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            sum_step_reward = 0 

        else:
            sum_step_reward += reward
            
        episode_reward += reward

    episode_score = env.game.score
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score
    
    snake_size = info["snake_size"] + 1
    print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}")
    terminal_length_distrib[snake_size] += 1
    total_reward += episode_reward
    total_score += env.game.score
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"=================== Summary ==================")
print(f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODE}")
print(terminal_length_distrib)
plt.bar(range(3, 443), terminal_length_distrib[3:443])
plt.show()
