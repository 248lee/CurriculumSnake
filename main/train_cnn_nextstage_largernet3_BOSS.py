import os
import sys
import random

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from trmaskppo_multiPolicy import TRMaskablePPOMultiPolicy
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_cnn import SnakeEnv

if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2
else:
    NUM_ENV = 32
LOG_DIR = "logs"
ExperimentName = "snake_ob_BOSS_please_success"

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
        env = SnakeEnv(seed=seed, length=state_name_list, formation='終焉', max_length=None, is_grow=True, silent_mode=True)
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
        model = TRMaskablePPOMultiPolicy(
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
        lr_schedule = linear_schedule(2.5e-4, 7.5e-5)
        # clip_range_schedule = linear_schedule(0.150, 0.025)
        import torch as th
        from network_structures import Stage2CustomFeatureExtractorCNN
        policy_kwargs = dict(
            features_extractor_class=Stage2CustomFeatureExtractorCNN,
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[512, 256, 128], vf=[256, 32])
        )

        directory = "./coaches"
        coaches_names = [filename[:len(filename) - 4] for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename))]
        policy_model_paths = ["coaches/" + cn for cn in coaches_names]
        mc_value_model_paths = ["trained_models_cnn/point985/mc_value_evaluation_" + cn  + "_in_BOSS" for cn in coaches_names]
        value_of_mc_policy_path = "trained_models_cnn/mc_value_evaluation_mc_policy_in_BOSS"
        print(policy_model_paths)
        print(mc_value_model_paths)
        from gamma import gamma
        model = TRMaskablePPOMultiPolicy(
            "CnnPolicy",
            env,
            policy_model_paths=policy_model_paths,
            mc_value_model_paths=mc_value_model_paths,
            value_of_mc_policy_path=value_of_mc_policy_path,
            device="cuda",
            verbose=1,
            n_steps=2048,
            batch_size=2048,
            n_epochs=4,
            gamma=gamma,
            learning_rate=lr_schedule,
            clip_range=9487,
            tensorboard_log=LOG_DIR,
            policy_kwargs=policy_kwargs
        )

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
        total_timesteps=int(200000000),
        callback=[checkpoint_callback],
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
