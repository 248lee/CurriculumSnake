import pickle
import random
import sys
import time

import pygame

from snake_game import SnakeGame


if __name__ == "__main__":

    seed = random.randint(0, 1e9)
    game = SnakeGame(seed=seed, length=3, is_grow=True, silent_mode=False)
    pygame.init()
    game.screen = pygame.display.set_mode((game.display_width, game.display_height))
    pygame.display.set_caption("Snake Game")
    game.font = pygame.font.Font(None, 36)
    

    game_state = "welcome"

    # Two hidden button for start and retry click detection
    start_button = game.font.render("START", True, (0, 0, 0))
    retry_button = game.font.render("RETRY", True, (0, 0, 0))

    update_interval = 0.15
    start_time = time.time()
    recording_path = "recordings/len441_2024_08_30_13_59_52.obj"
    with open(recording_path, 'rb') as file:
        recording = pickle.load(file)
        snake_record = recording['snake_record']
        food_record = recording['food_record']

    i = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if game_state == "welcome" and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(start_button):
                    for i in range(3, 0, -1):
                        game.screen.fill((0, 0, 0))
                        game.draw_countdown(i)
                        # game.sound_eat.play()
                        pygame.time.wait(1000)
                    action = -1  # Reset action variable when starting a new game
                    game_state = "running"
                    i = 0

            if game_state == "game_over" and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(retry_button):
                    for i in range(3, 0, -1):
                        game.screen.fill((0, 0, 0))
                        game.draw_countdown(i)
                        # game.sound_eat.play()
                        pygame.time.wait(1000)
                    game.reset()
                    action = -1  # Reset action variable when starting a new game
                    game_state = "running"
        
        if game_state == "welcome":
            game.draw_welcome_screen()

        if game_state == "game_over":
            game.draw_game_over_screen()
            i = 0

        if game_state == "running":
            game.snake = snake_record[i].copy()
            game.food = food_record[i]
            game.direction = "WAITING"
            if game.snake[0][0] - game.snake[1][0] == 1:  # if head is lower than snake's first body
                game.direction = "DOWN"
            elif game.snake[0][0] - game.snake[1][0] == -1:  # if head is higher than snake's first body
                game.direction = "UP"
            elif game.snake[0][1] - game.snake[1][1] == 1:  # if head is righter than snake's first body
                game.direction = "RIGHT"
            elif game.snake[0][1] - game.snake[1][1] == -1:  # if head is lefter than snake's first body
                game.direction = "LEFT"
            game.render()
            i += 1
            print(i)

            if i == len(snake_record):
                game_state = "game_over"
    
        time.sleep(0.05)
