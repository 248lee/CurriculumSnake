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
    MODEL_PATH = r"trained_models_cnn/snake21_len300_please_success_38000000_steps.zip"

NUM_EPISODE = 300

RENDER = True
FRAME_DELAY = 0.01 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5
VALUE_MODEL_NAME = None# {"actor": "trained_models_cnn/snake_s7_l4_grow_g985_160000000_steps.zip", "critic": "trained_models_value/DVN_transfer_s7toBOSS_final.zip"}
VALUE_MODEL_NAME2 = None# {"actor": "trained_models_cnn/snake_s7_l4_grow_g985_160000000_steps.zip", "critic": "trained_models_value/DVN_transfer_s7toBOSS_unfreeze_test_final.zip"}
AC_MODEL_NAME = None# "trained_models_cnn/snake_s7_l4_grow_g985_160000000_steps"

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

state_name_list = [
    'len350_state_2024_08_15_07_51_07.obj',
    'len352_state_2024_08_15_07_46_41.obj',
    'len353_state_2024_08_15_08_46_43.obj',
    'len356_state_2024_08_15_08_59_36.obj',
    'len358_state_2024_08_15_08_47_30.obj',
    'len359_state_2024_08_15_09_00_32.obj',
    'len366_state_2024_08_15_08_48_32.obj',
    'len369_state_2024_08_15_08_49_35.obj'
]

if RENDER:
    env = SnakeEnv(seed=seed, length = state_name_list, is_grow=True, limit_step=True, silent_mode=False)
else:
    env = SnakeEnv(seed=seed, length = state_name_list, is_grow=True, limit_step=True, silent_mode=True)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)
if VALUE_MODEL_NAME != None:
    value_model = DVNNetwork(old_model_name=VALUE_MODEL_NAME["actor"]).to('cuda')
    value_model.load_state_dict(th.load(VALUE_MODEL_NAME["critic"]))
    value_model.eval()
if VALUE_MODEL_NAME2 != None:
    value_model2 = DVNNetwork(old_model_name=VALUE_MODEL_NAME2["actor"]).to('cuda')
    value_model2.load_state_dict(th.load(VALUE_MODEL_NAME2["critic"]))
    value_model2.eval()
if AC_MODEL_NAME != None:
    ac_model = MaskablePPO.load(AC_MODEL_NAME)
    ac_model.policy.set_training_mode(False)

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
        # if info != None and info["snake_size"] >= 350:
        #     is_save = input()
        #     if is_save == 's':
        #         env.save_state()

        model.policy.set_training_mode(False)
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        if VALUE_MODEL_NAME != None or VALUE_MODEL_NAME2 != None:
            obs, _ = model.policy.obs_to_tensor(obs)
        if VALUE_MODEL_NAME != None:
            print(VALUE_MODEL_NAME["actor"] + "'s policy value by TD:", value_model(obs).item())
        if VALUE_MODEL_NAME2 != None:
            print(VALUE_MODEL_NAME2["actor"] + "'s policy value by TD:", value_model2(obs).item())
        if AC_MODEL_NAME != None:
            print(AC_MODEL_NAME+ "'s policy value by MC:", ac_model.policy.predict_values(obs).item())
        if VALUE_MODEL_NAME != None or VALUE_MODEL_NAME2 != None or AC_MODEL_NAME != None:
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
