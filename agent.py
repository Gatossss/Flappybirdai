import torch
import random
import numpy as np
from collections import deque
from flappybird import FlappyBird
from model import Linear_QNet, QTrainer
from helper import plot

import matplotlib.pyplot as plt
import time
import networkx as nx
from torch import nn

MAX_MEMORY = 10000000
BATCH_SIZE = 64
LR = 0.00001
plt.ion()


def get_device():
    device = torch.device("cpu")
    print('Using CPU')
    return device


class Agent:
    def __init__(self):
        self.device = get_device()
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(6, 64, 2, self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # Wrap the model with nn.DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

    def get_state(self, game):
        height = game.birdY
        bottom_pipe_height = game.downRect.y
        upper_pipe_height = game.upRect.y
        jump = game.jump
        jumpspeed = game.jumpSpeed
        velocity = game.gravity

        state = [
            bottom_pipe_height,
            upper_pipe_height,
            height,
            jump,
            jumpspeed,
            velocity
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

        self.trainer.train_step

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state,):
        #random moves lol
        self.epsilon = 10
        final_move = [0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
        


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = FlappyBird()
    while True:
        time.sleep(0.0166666)
        #get old state
        game.birdUpdate() 
        state_old = agent.get_state(game)
        #get move
        final_move = agent.get_action(state_old)
        action = final_move
        #perform move and get new state
        reward, done, counter = game.run(final_move)

            
        state_new = agent.get_state(game)
        #train short mem
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        #remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            #train long mem, plot result?
            agent.n_games += 1
            agent.train_long_memory()
            if counter > record:
                record = counter
                agent.model.save()
            print("Game: ", agent.n_games, "Score: ", counter, "Record: ", record)
            plot_scores.append(counter)
            total_score += counter
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
            
            game.reset()


if __name__ == "__main__":
    train()