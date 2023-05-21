#!/usr/bin/env python

import pygame
from pygame.locals import *  # noqa
import sys
import random
import numpy as np
from helper import plot


class FlappyBird:
    def __init__(self):
        self.screen = pygame.display.set_mode((400, 708))
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha(),
                            pygame.image.load("assets/dead.png")]
        self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("assets/top.png").convert_alpha()
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 10
        self.gravity = 5
        self.dead = False
        self.sprite = 0
        self.counter = 0
        self.offset = random.randint(-110, 110)
        self.done = False
        self.reward = 0
        self.frame_iteration = 0
        self.gap = 130 

    def updateWalls(self):
        self.reward = 0
        self.wallx -= 2
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            self.reward = 10
            self.offset = random.randint(-110, 110)
        return self.reward, self.counter

    def birdUpdate(self):
        counter = self.counter
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())
        self.downRect = downRect
        self.upRect = upRect
        self.reward = 0
        if upRect.colliderect(self.bird):
            self.dead = True
            self.reward = -10
            self.done = True
        if downRect.colliderect(self.bird):
            self.dead = True
            self.reward = -10
            self.done = True
        if not 0 < self.bird[1] < 720:
            self.reward = -10
            self.done = True
        return self.downRect, self.upRect, counter

    def reset(self):
        self.bird[1] = 708 // 2 - self.bird.height // 2
        self.birdY = 708 // 2 - self.bird.height // 2
        self.dead = False
        self.counter = 0
        self.wallx = 400
        self.offset = random.randint(-110, 110)
        self.gravity = 5
        self.done = False

    def run(self, action):
        BirdY = self.birdY
        if BirdY < self.upRect.top and BirdY > self.downRect.bottom:
            self.reward = 0.1
        placeholder = 0
        if placeholder == 0:
            if self.counter == 0:
                self.gap = 400
            if self.counter == 1:
                self.gap = 350
            if self.counter == 2:
                self.gap = 300
            if self.counter == 3:
                self.gap = 250
            if self.counter == 4:
                self.gap = 200
            if self.counter == 5:
                self.gap = 150
            if self.counter == 6:
                self.gap = 130


        reward = self.reward
        done = self.done
        counter = self.counter
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        clock.tick(6000000)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break  # Break out of the for loop
        if np.array_equal(action, [0,1]):
            self.jump = 17
            self.gravity = 5
            self.jumpSpeed = 10
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.wallUp,
                         (self.wallx, 360 + self.gap - self.offset))
        self.screen.blit(self.wallDown,
                         (self.wallx, 0 - self.gap - self.offset))
        self.screen.blit(font.render(str(self.counter),
                                     -1,
                                     (255, 255, 255)),
                         (200, 50))
        if self.dead:
            self.sprite = 2
        elif self.jump:
            self.sprite = 1
        self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
        if not self.dead:
            self.sprite = 0
        self.updateWalls()
        self.birdUpdate()
        pygame.display.update()
        return reward, done, counter
