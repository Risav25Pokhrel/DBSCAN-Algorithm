import pygame as pg
import numpy as np
import pandas as pd

###################################################
# Author: Rohan Bade
# github: https://github.com/RohanBade
####################################################


# Define some colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Extended COLOR_MAP for more clusters
COLOR_MAP = {
    -2: WHITE,  # Unprocessed
    -1: RED,    # Noise
    0: YELLOW,
    1: BLUE,
    2: GREEN,
    3: (255, 165, 0),  # Orange
    4: (75, 0, 130),   # Indigo
    5: (255, 192, 203),# Pink
    6: (0, 255, 255),  # Cyan
    7: (128, 0, 128),  # Purple
    8: (255, 215, 0),  # Gold
    9: (0, 128, 0),    # Dark Green
    10: (128, 128, 0)  # Olive
}

class MyDBSCAN:
    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts
        self.data=np.array([
            [0.1, 1.0],
            [0.2, 0.9],
            [0.3, 1.0],
            [0.4, 0.6],
            [0.5, 0.6],
            [0.6, 0.5],
            [0.7, 0.8],
            [0.8, 0.1],
            [0.9, 0.2],
            [1.0, 0.1]
        ])
        # Initialize Pygame
        pg.init()
        self.screen_size = (480, 480)
        self.screen = pg.display.set_mode(self.screen_size)
        pg.display.set_caption("DBSCAN Algorithm Simulation")
        self.clock = pg.time.Clock()
        self.running = True

    def draw_grid(self):
        self.screen.fill(BLACK)
        block_size = 40
        for x in range(40, 440, block_size):
            for y in range(40, 440, block_size):
                rect = pg.Rect(x, y, block_size, block_size)
                pg.draw.rect(self.screen, WHITE, rect, 1)
        self.draw_points()
        pg.display.flip()

    def draw_points(self):
        for index, point in enumerate(self.data):
            pg.draw.circle(self.screen, COLOR_MAP.get(self.labels[index]), (point * 400 + 40).astype(int), 10)

    def cluster(self):
        self.labels = np.full(len(self.data), -2)  # -2 means unprocessed

        c_id = -1
        for i in range(len(self.data)):
            if self.labels[i] != -2:
                continue

            neighbors = self.region_query(i)
            if len(neighbors) < self.min_pts:
                self.labels[i] = -1  # noise
            else:
                c_id += 1
                self.expand_cluster(i, neighbors, c_id)

            # Redraw screen
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_points()

            # Update display
            pg.display.flip()

            # Slow down the process
            self.clock.tick(1)

            # Check for events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                    return

        self.running = False

    def expand_cluster(self, p, neighbors, c_id):
        self.labels[p] = c_id
        i = 0
        while i < len(neighbors):
            pn = neighbors[i]
            if self.labels[pn] == -1:
                self.labels[pn] = c_id
            elif self.labels[pn] == -2:
                self.labels[pn] = c_id
                new_neighbors = self.region_query(pn)
                if len(new_neighbors) >= self.min_pts:
                    neighbors = neighbors + new_neighbors
            i += 1

    def euc_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def region_query(self, p):
        result = []
        for i in range(len(self.data)):
            if self.euc_distance(self.data[p], self.data[i]) < self.eps:
                result.append(i)
        return result

    def run(self):
        self.labels = np.full(len(self.data), -2)
        current_index = 0
        c_id = -1
        self.screen.fill(BLACK)
        self.draw_grid()
        self.draw_points()
        pg.display.flip()

        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False

            if current_index < len(self.data):
                if self.labels[current_index] == -2:
                    self.draw_grid()
                    pg.time.wait(1000)
                    neighbors = self.region_query(current_index)
                    if len(neighbors) < self.min_pts:
                        self.labels[current_index] = -1
                        self.draw_grid()
                        pg.time.wait(1000)
                    else:
                        c_id += 1
                        self.expand_cluster(current_index, neighbors, c_id)
                current_index += 1

                print("yes")
                # Redraw screen
                self.draw_grid()
                self.clock.tick(1)  # Slow down the process to 1 FPS

        pg.quit()

if __name__ == "__main__":
    dbscan = MyDBSCAN(0.2, 2)
    dbscan.run()
