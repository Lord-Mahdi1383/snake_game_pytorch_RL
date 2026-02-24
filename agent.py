import torch
import random
import numpy as np
from collections import deque

from snake_game_pygame import snake, UP, DOWN, LEFT, RIGHT
from model import DQN, DQN_trainer
from plotter import plot



class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100000)

        self.model = DQN(11, 256, 3)
        self.trainer = DQN_trainer(self.model, lr=0.005, gamma=self.gamma)

    
    def get_state(self, game):
        head = game.snake[0]

        # points around snake head 
        point_L = [head[0] - 20, head[1]]
        point_R = [head[0] + 20, head[1]]
        point_U = [head[0], head[1] - 20]
        point_D = [head[0], head[1] + 20]

        # current direction
        dir_L = game.direction == LEFT
        dir_R = game.direction == RIGHT
        dir_U = game.direction == UP
        dir_D = game.direction == DOWN

        state = [
            # danger ahead
            (dir_R and game.collision(point_R)) or
            (dir_L and game.collision(point_L)) or
            (dir_U and game.collision(point_U)) or
            (dir_D and game.collision(point_D)),

            # danger right
            (dir_R and game.collision(point_D)) or
            (dir_L and game.collision(point_U)) or
            (dir_U and game.collision(point_R)) or
            (dir_D and game.collision(point_L)),

            # danger left 
            (dir_R and game.collision(point_U)) or
            (dir_L and game.collision(point_D)) or
            (dir_U and game.collision(point_L)) or
            (dir_D and game.collision(point_R)),

            # move dir
            dir_L,
            dir_R,
            dir_U,
            dir_D,

            # food loc (is a list of [x,y])
            game.food[0] < game.head[0],
            game.food[0] > game.head[0],
            game.food[1] < game.head[1],
            game.food[1] > game.head[1] 
        ]
        return np.array(state, dtype=int)
     
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train_long_memory(self):
        if len(self.memory) > 1000:
            samp = random.sample(self.memory, 1000)
        else:
            samp = self.memory

        states, actions, rewards, next_states, dones = zip(*samp)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
            final_move[move] = 1

        return final_move
    

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = snake()

    while True:
        old_state = agent.get_state(game)
        final_move = agent.get_action(old_state)
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        agent.train_short_memory(old_state, final_move, reward, new_state, done)
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
