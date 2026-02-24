import pygame
import random
import numpy as np

# constants
SPEED = 1000
BLOCK_SIZE = 20

RIGHT = 1
LEFT = 2
UP = 3
DOWN = 4

window_x = 720
window_y = 480

black = pygame.Color(0,0,0)
white = pygame.Color(255,255,255)
red = pygame.Color(255,0,0)
green = pygame.Color(0,255,0)
blue = pygame.Color(0,0,255)

pygame.init()
font = pygame.font.SysFont('arial', 20)


class snake:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        self.direction = RIGHT

        self.head = (self.w//2, self.h//2)
        
        self.snake = [self.head,
                      (self.head[0]-BLOCK_SIZE, self.head[1]),
                      (self.head[0]-(2*BLOCK_SIZE), self.head[1])]
        
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0


    def place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = (x, y)

        if self.food in self.snake:
            self.place_food()


    def play_step(self, action):
        self.frame_iteration += 1

        # user input 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # move 
        self.move(action)
        self.snake.insert(0, self.head)

        # game over check
        reward = 0
        game_over = False
        if self.collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # new food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()


        # ui and time
        self.update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score


    def collision(self, point=None):
        if point is None:
            point = self.head
    
        # hits wall 
        if point[0] > self.w - BLOCK_SIZE or point[0] < 0 or point[1] > self.h - BLOCK_SIZE or point[1] < 0:
            return True
        
        # hits itself
        if point in self.snake[1:]:
            return True
        
        return False


    def update_ui(self):
        self.display.fill(black)

        for point in self.snake:
            pygame.draw.rect(self.display, blue, pygame.Rect(point[0], point[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, green, pygame.Rect(point[0]+4, point[1]+4, 12, 12))
        
        pygame.draw.rect(self.display, red, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        txt = font.render("Score: " + str(self.score), True, white)
        self.display.blit(txt, [0,0])
        pygame.display.flip()

    def move(self, action):
        clock_wize = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wize.index(self.direction)

        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wize[idx]

        elif np.array_equal(action, [0,1,0]):
            new_idx = (idx + 1) % 4
            new_dir = clock_wize[new_idx]

        else:
            new_idx = (idx - 1) % 4
            new_dir = clock_wize[new_idx]


        self.direction = new_dir

        x = self.head[0]
        y = self.head[1]

        if self.direction == RIGHT:
            x += BLOCK_SIZE
        elif self.direction == LEFT:
            x -= BLOCK_SIZE
        elif self.direction == UP:
            y -= BLOCK_SIZE
        elif self.direction == DOWN:
            y += BLOCK_SIZE    


        self.head = (x,y)

