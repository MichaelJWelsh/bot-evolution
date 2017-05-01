"""
Bot Evolution v1.0.0
"""

import os
import sys
import pickle
import pygame as pg
from pygame.locals import *
import numpy as np
import datetime
import settings
import population

def main():
    np.random.seed()
    pg.init()

    # Initialize runtime variables.
    periodically_save = False
    pop = None
    if os.path.isfile("save.txt") and input("Save file detected! Use it? (y/n): ").lower() == 'y':
        settings.FPS, settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT, settings.TIME_MULTIPLIER, pop = pickle.load(open("save.txt", "rb"))
    else:
        pop_size = 0
        mutation_rate = 0
        while True:
            pop_size = int(input("Population size: "))
            if pop_size < 5:
                print("Population size must be at least 5!")
            else:
                break
        while True:
            mutation_rate = float(input("Mutation rate: "))
            if mutation_rate <= 0 or mutation_rate >= 1:
                print("Mutation rate must be in the range (0, 1)!")
            else:
                break
        while True:
            settings.TIME_MULTIPLIER = float(input("Time multiplier: "))
            if settings.TIME_MULTIPLIER < 1:
                print("Time multiplier must be at least 1!")
            else:
                break
        if input("Advance options? (y/n): ").lower() == 'y':
            while True:
                settings.FPS = int(input("Frames per second: "))
                if settings.FPS < 1:
                    print("FPS must be at least 1!")
                else:
                    break
            while True:
                settings.WINDOW_WIDTH = int(input("Window width: "))
                if settings.WINDOW_WIDTH < 50:
                    print("Window width must be at least 50!")
                else:
                    break
            while True:
                settings.WINDOW_HEIGHT = int(input("Window height: "))
                if settings.WINDOW_HEIGHT < 50:
                    print("Window height must be at least 50!")
                else:
                    break
        pop = population.Population(pop_size, mutation_rate)
    if input("Periodically save every half hour? (y/n): ").lower() == 'y':
        periodically_save = True
    print("\nNote: ")
    print("\tPress 'r' to reset the population.")
    print("\tPress 'p' to pause / unpause.")
    print("\tPress 's' to save population's data (for use next time).")
    print("\tPress 'up' / 'down' to change the populations mutation rate.")
    print("\tPress 'left' / 'right' to change the time multiplier.")
    print("\tClick on the screen to lay down food.")

    # Core variables.
    FONT_SIZE = 30
    FONT = pg.font.SysFont("Arial", FONT_SIZE)
    fps_clock = pg.time.Clock()
    window = pg.display.set_mode((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT))
    pg.display.set_caption("Bot Evolution")

    # Main loop.
    dt = 0.0
    fps_clock.tick(int(settings.FPS / (settings.TIME_MULTIPLIER / 5.0 + 1)))
    paused = False
    while True:
        key_pressed = {"up": False, "down": False, "left": False, "right": False}
        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                sys.exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_r:
                    pop = population.Population(pop.SIZE, pop.mutation_rate)
                if event.key == pg.K_s:
                    pickle.dump([settings.FPS, settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT, settings.TIME_MULTIPLIER, pop], open("save.txt", "wb"))
                if event.key == pg.K_p:
                    paused = not paused
            elif event.type == pg.MOUSEBUTTONUP:
                pos = pg.mouse.get_pos()
                pop.food.pop()
                food = population.Food(pop)
                food.x = pos[0]
                food.y = pos[1]
                pop.food.append(food)
        if paused:
            dt = fps_clock.tick(int(settings.FPS / (settings.TIME_MULTIPLIER / 5.0 + 1))) / 1000.0 * int(settings.FPS / (settings.TIME_MULTIPLIER / 5.0 + 1))
            continue
        if periodically_save and datetime.datetime.now().minute % 30 == 0:
            pickle.dump([settings.FPS, settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT, settings.TIME_MULTIPLIER, pop], open("save.txt", "wb"))
        keys = pg.key.get_pressed()
        if keys[pg.K_UP]:
            key_pressed["up"] = True
        if keys[pg.K_DOWN]:
            key_pressed["down"] = True
        if key_pressed["up"] and key_pressed["down"]:
            key_pressed["up"] = False
            key_pressed["down"] = False
        if keys[pg.K_LEFT]:
            key_pressed["left"] = True
        if keys[pg.K_RIGHT]:
            key_pressed["right"] = True
        if key_pressed["left"] and key_pressed["right"]:
            key_pressed["left"] = False
            key_pressed["right"] = False
        update(dt, pop, key_pressed)
        window.fill((0, 0, 0))
        render(window, FONT, pop)
        pg.display.update()
        dt = fps_clock.tick(int(settings.FPS / (settings.TIME_MULTIPLIER / 5.0 + 1))) / 1000.0 * int(settings.FPS / (settings.TIME_MULTIPLIER / 5.0 + 1))

display_time_remaining = 0.0
def update(dt, pop, key_pressed):
    global display_time_remaining
    if key_pressed["up"] or key_pressed["down"] or key_pressed["left"] or key_pressed["right"]:
        display_time_remaining = 3.0

        if key_pressed["up"]:
            pop.mutation_rate += 0.001
        elif key_pressed["down"]:
            pop.mutation_rate -= 0.001
        if pop.mutation_rate <= 0:
            pop.mutation_rate = 0.001
        elif pop.mutation_rate >= 1:
            pop.mutation_rate = 0.999

        if key_pressed["left"]:
            settings.TIME_MULTIPLIER -= 0.1
        elif key_pressed["right"]:
            settings.TIME_MULTIPLIER += 0.1
        if settings.TIME_MULTIPLIER < 1:
            settings.TIME_MULTIPLIER = 1.0
    else:
        display_time_remaining -= 1.0 / settings.FPS * dt
        if display_time_remaining < 0:
            display_time_remaining = 0.0

    pop.update(dt)

def render(window, FONT, pop):
    for food in pop.food:
        pg.draw.circle(window, food.RGB, (int(food.x), int(food.y)), food.HITBOX_RADIUS)

    for bot in pop.bots:
        # Draw body.
        pg.draw.circle(window, bot.RGB, (int(bot.x), int(bot.y)), bot.HITBOX_RADIUS)

        # Draw field-of-vision lines.
        LINE_THICKNESS = 1
        PROTRUSION = int(bot.HITBOX_RADIUS * 1.5)
        to_x = int(bot.x + (bot.HITBOX_RADIUS + PROTRUSION) * np.cos(bot.theta - bot.FIELD_OF_VISION_THETA / 2))
        to_y = int(bot.y - (bot.HITBOX_RADIUS + PROTRUSION) * np.sin(bot.theta - bot.FIELD_OF_VISION_THETA / 2))
        pg.draw.line(window, bot.RGB, (bot.x, bot.y), (to_x, to_y), LINE_THICKNESS)
        to_x = int(bot.x + (bot.HITBOX_RADIUS + PROTRUSION) * np.cos(bot.theta + bot.FIELD_OF_VISION_THETA / 2))
        to_y = int(bot.y - (bot.HITBOX_RADIUS + PROTRUSION) * np.sin(bot.theta + bot.FIELD_OF_VISION_THETA / 2))
        pg.draw.line(window, bot.RGB, (bot.x, bot.y), (to_x, to_y), LINE_THICKNESS)

    if display_time_remaining > 0:
        resultSurf = FONT.render("Mutation Rate: %.3f       Speed: %.1fx" % (pop.mutation_rate, settings.TIME_MULTIPLIER), True, (255, 255, 255))
        resultRect = resultSurf.get_rect()
        resultRect.topleft = (25, 25)
        window.blit(resultSurf, resultRect)

if __name__ == "__main__":
    main()
