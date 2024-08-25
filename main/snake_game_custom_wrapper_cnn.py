import math

import gymnasium as gym
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env):
    def __init__(self, length, is_grow, max_length = None, seed=0, board_size=21, silent_mode=True, limit_step=True, random_states=[]):
        super().__init__()
        self.game = SnakeGame(length, is_grow, seed=seed, board_size=board_size, silent_mode=silent_mode, random_states=random_states)
        self.game.reset()

        self.silent_mode = silent_mode

        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(board_size, board_size, 3),
            dtype=np.uint8
        )

        self.board_size = board_size
        self.grid_size = board_size ** 2 # Max length of snake is board_size^2
        if max_length == None:
            self.max_length = self.grid_size
        else:
            self.max_length = max_length
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        if limit_step:
            self.step_limit = self.grid_size * 2 # More than enough steps to get the food.
        else:
            self.step_limit = 1e9 # Basically no limit.
        self.reward_step_counter = 0
    
    def seed(self, arg1):
        pass

    def save_state(self):
        self.game.save_state()

    def load_state(self, file_name):
        self.game.load_state(file_name)
        self.done = False
        self.reward_step_counter = 0
        self.total_steps = 0
        obs = self._generate_observation()
        return obs, None
    
    def reset(self, seed, options):
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0
        self.total_steps = 0

        obs = self._generate_observation()
        return obs, None
    
    def step(self, action):
        self.done, info = self.game.step(action) # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()
        reward = 0.0
        self.reward_step_counter += 1

        if info["snake_size"] == self.max_length: # Snake fills up the entire board. Game over.
            # reward = self.max_growth * 0.1
            reward = 0.5
            self.done = True
            if not self.silent_mode:
                self.render()
            #     self.game.sound_victory.play()
            return obs, reward, self.done, False, info
        
        if self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True

        truncate = False
        # if self.total_steps > 5 * self.step_limit:
        #     truncate = True
        
        if self.done: # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            # reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)            
            # reward = reward * 0.002  # original: * 0.1
            reward = -(info["snake_size"] + 66) * 2 / self.grid_size
            return obs, reward, self.done, False, info
          
        elif info["food_obtained"]: # Food eaten. Reward boost on snake size.
            reward = (info["snake_size"] + 66) / self.grid_size
            reward = reward * 10
            self.reward_step_counter = 0 # Reset reward step counter
        
        else:
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 2 / info["snake_size"]
            else:
                reward = -2 / info["snake_size"]
            if info['snake_size'] >= 100:
                reward = -self.reward_step_counter * 0.00005  # if the snake is long enough, it shouldn't get step reward
            else:
                reward = reward * 0.1

        # max_score: 72 + 14.1 = 86.1
        # min_score: -14.1

        if not self.silent_mode:
            self.render()
            import time
            FRAME_DELAY = 0.05
            time.sleep(FRAME_DELAY)

        reward *= 0.01  # smallalize the reward
        self.total_steps += 1
        return obs, reward, self.done, truncate, info
    
    def render(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                (row, col) in snake_list # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)
        
        # Stack single layer into 3-channel-image.
        obs = np.stack((obs, obs, obs), axis=-1)
        
        # Set the snake head to green and the tail to blue
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # Set the food to red
        obs[self.game.food] = [0, 0, 255]

        # Enlarge the observation x7
        # obs = np.repeat(np.repeat(obs, 4, axis=0), 4, axis=1)

        return obs

# Test the environment using random actions
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(silent_mode=False)
    
    # # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     done = False
    #     i = 0
    #     while not done:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()
            
    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)
    
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
