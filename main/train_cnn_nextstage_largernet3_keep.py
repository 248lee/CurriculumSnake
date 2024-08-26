import os
import sys
import random

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_cnn import SnakeEnv

if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2
else:
    NUM_ENV = 32
LOG_DIR = "logs"
ExperimentName = "snake21_len280_max380"

os.makedirs(LOG_DIR, exist_ok=True)

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(seed=0):
    def _init():
        # Specify the directory
        directory = "./game_states"

        # Get the list of filenames in the specified directory
        state_name_list = [filename for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename))]
        state_name_list = [
            "len277_state_2024_08_18_17_34_09.obj",
            "len284_state_2024_08_18_17_34_21.obj",
            "len285_state_2024_08_25_15_19_10.obj",
            "len286_state_2024_08_18_17_34_32.obj",
            "len289_state_2024_08_25_15_20_12.obj",
            "len291_state_2024_08_25_15_21_01.obj",
        ]
        env = SnakeEnv(seed=seed, length=state_name_list, max_length=380, is_grow=True, silent_mode=True)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def main():

    # Generate a list of random seeds for each environment.
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])

    if torch.backends.mps.is_available():
        lr_schedule = linear_schedule(5e-4, 2.5e-6)
        # clip_range_schedule = linear_schedule(0.150, 0.025)
        # Instantiate a PPO agent using MPS (Metal Performance Shaders).
        model = TRMaskablePPO(
            "CnnPolicy",
            # env,
            # old_model_name="trained_models_cnn/snake_s1_len3_9000000_steps",
            # dvn_model_name="trained_models_value/DVN_transfer_final.zip",
            # device="mps",
            # verbose=1,
            # n_steps=2048,
            # batch_size=512*8,
            # n_epochs=4,
            # gamma=0.94,
            # learning_rate=lr_schedule,
            # clip_range=clip_range_schedule,
            # tensorboard_log=LOG_DIR
        )
    else:
        lr_schedule = 2.5e-6
        # clip_range_schedule = linear_schedule(0.150, 0.025)
        custom_objects = {
        "learning_rate": lr_schedule,
        "gamma": 0.985,
        "clip_range": 0.002
        }
        model = MaskablePPO.load("trained_models_cnn/snake21_len280_max380_130000000_steps", env=env,custom_objects=custom_objects)

    # Set the save directory
    if torch.backends.mps.is_available():
        save_dir = "trained_models_cnn_mps"
    else:
        save_dir = "trained_models_cnn"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 15625 * 4 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix=ExperimentName)

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")

    model.learn(
        total_timesteps=int(50000000),
        callback=[checkpoint_callback],
        reset_num_timesteps=False,
        tb_log_name=ExperimentName,
        progress_bar=True
    )
    env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, ExperimentName + ".zip"))

if __name__ == "__main__":
    main()
