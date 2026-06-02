#Training my first RL bot with animation
import matplotlib.pyplot as plt
import pygame
import numpy as np
import random
import pickle
import os 
import time

class maze():
    def __init__ (self, maze, start_position, end_position):
        self.maze = maze
        self.col = len(maze[0])
        self.row = len(maze)
        self.start_position = tuple(start_position)
        self.end_position = tuple(end_position)
        self.current_position = tuple(self.start_position)
        self.rewards = []
        self.rewards_sarsa = []

    def __repr__(self):
        print("This is some information about me", self.maze, "row", self.row,"col",  self.col, "Current position", self.current_position)
        return ""

    def reset(self):
        self.current_position = self.start_position
        return (self.current_position)

    
    def is_valid_position(self, r, c):
        if not (0 <= r < self.row and 0 <= c < self.col):
            return False # Out of bounds
        if self.maze[r][c] == 1:
            return False # Hit a wall
        return True
    
    def step(self, action):
        reward = 0
        #0 -> Up, 1 -> Down 2 -> Left 3 -> Right
        action_map = {0: (1, 0),
                      1: (-1, 0),
                      2: (0, -1),
                      3: (0, 1)}
        if action not in action_map:
            return 
        
        r, c = self.current_position
        dr, dc = action_map[action]
        new_r = r + dr
        new_c = c + dc

        new_position = (r, c)
        reward = -1
        done = False

        if self.is_valid_position(new_r, new_c):
            new_position = (new_r, new_c)
            self.current_position = new_position

            if new_position == tuple(self.end_position):
                reward = 10
                done = True
        else:
            reward = -10

        return (new_position, reward, done)

def plot_expected_reward(x, y):

    plt.plot(x, y)
def draw_maze(screen, maze_obj, current_pos=None, font=None, size=50):
    
    """Draw the maze with optional current position highlight"""
    screen.fill((128, 128, 128))  # Gray background
    
    for r in range(maze_obj.row):
        for c in range(maze_obj.col):
            x = 200 + c * size
            y = 500 - r * size
            rect = pygame.Rect(x, y, size, size)
            
            # Choose color based on cell type
            if (r, c) == maze_obj.start_position:
                color = (0, 255, 0)  # Green for start
            elif (r, c) == maze_obj.end_position:
                color = (255, 0, 0)  # Red for end
            elif maze_layout[r][c] == 1:
                color = (255, 255, 255)  # White for walls
            else:
                color = (255, 255, 0)  # Yellow for open spaces

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 2)  # Black border
            
            # Draw current position
            if current_pos and (r, c) == current_pos:
                pygame.draw.circle(screen, (0, 0, 255), rect.center, 15)  # Blue circle for agent

            # Draw coordinates
            if font:
                label = f"[{r}, {c}]"
                text_surf = font.render(label, True, (0, 0, 0))
                text_rect = text_surf.get_rect(center=(rect.centerx, rect.bottom - 10))
                screen.blit(text_surf, text_rect)

def train_using_sarsa():
    maze_layout = [[0] * 5 for n in range(3)]
    maze_layout[0][2] = 1
    maze_layout[1][2] = 1
    start = [0, 0]
    end = [0, 4]
    new_maze = maze(maze_layout, start, end)

    Q_TABLE = {}
    ACTIONS = [0, 1, 2, 3]

    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EPSILON = 1.0
    EPSILON_DECAY_RATE = 0.999
    MIN_EPSILON = 0.01
    EPISODES = 2000

    for r in range(new_maze.row):
        for c in range(new_maze.col):
            Q_TABLE[(r, c)] = {action: 0 for action in ACTIONS}



    for i in range(EPISODES):

        state = new_maze.reset()
        steps = 0
        total_reward = 0
        done = False
        current_epsilon = max(MIN_EPSILON, EPSILON * (EPSILON_DECAY_RATE ** i))

        while not done:
            if random.random() < current_epsilon:
                action = random.choice(ACTIONS)
            else:
                action = max(Q_TABLE[state], key=Q_TABLE[state].get)

            new_state, reward, done = new_maze.step(action)

            #continues same strategy of exploration
            if not done: 
                if random.random() < current_epsilon:
                    next_action = random.choice(ACTIONS)
                else:
                    next_action = max(Q_TABLE[new_state], key=Q_TABLE[new_state].get)

                # print("FAIL", Q_TABLE)
                next_q = Q_TABLE[new_state][next_action]

            else:
                next_q = 0

            old_q_value = Q_TABLE[state][action]
            new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_q - old_q_value)
            Q_TABLE[state][action] = new_q_value

            state = new_state
            action = next_action if not done else None

            total_reward += reward
        new_maze.rewards_sarsa.append(total_reward)
    return (Q_TABLE, new_maze)

def train_with_full_visualization():
    # Initialize maze
    maze_layout = [[0] * 5 for n in range(3)]
    maze_layout[0][2] = 1
    maze_layout[1][2] = 1
    start = [0, 0]
    end = [0, 4]
    new_maze = maze(maze_layout, start, end)
    
    # Initialize Q-table
    Q_TABLE = {}
    ACTIONS = [0, 1, 2, 3]
    
    for r in range(new_maze.row):
        for c in range(new_maze.col):
            Q_TABLE[(r, c)] = {action: 0 for action in ACTIONS}
    
    # Continue training without visualization for remaining episodes
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EPSILON = 1.0
    EPSILON_DECAY_RATE = 0.999
    MIN_EPSILON = 0.01
    EPISODES = 2000
    
    for episode in range(0, EPISODES): 
        total_reward = 0
        state = new_maze.reset()
        done = False
        current_epsilon = max(MIN_EPSILON, EPSILON * (EPSILON_DECAY_RATE ** episode))
        
        while not done:
            if random.random() < current_epsilon:
                action = random.choice(ACTIONS)
            else:
                action = max(Q_TABLE[state], key=Q_TABLE[state].get)
            
            new_state, reward, done = new_maze.step(action)
            
            old_q_value = Q_TABLE[state][action]
            next_max_q = max(Q_TABLE[new_state].values())
            new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q_value)
            Q_TABLE[state][action] = new_q_value
            
            state = new_state
            total_reward += reward
        
        new_maze.rewards.append(total_reward)
        # if episode % 100 == 0:
            #Display some nice pictures.
            # print(f"This is episode {episode} and this is the Q_Table: {Q_TABLE}\n")
            

    
    return Q_TABLE, new_maze

def show_final_policy(q_table, maze_obj):
    """Show the final learned policy"""
    pygame.init()
    screen = pygame.display.set_mode((750, 750))
    pygame.display.set_caption("Final Learned Policy")
    font = pygame.font.Font(None, 24)
    
    # Draw the maze
    draw_maze(screen, maze_obj, font=font)
    
    # Draw arrows showing the policy
    arrow_map = {
        0: (0, -1),  # Up
        1: (0, 1),   # Down
        2: (-1, 0),  # Left
        3: (1, 0)    # Right
    }
    
    size = 50
    for r in range(maze_obj.row):
        for c in range(maze_obj.col):
            current_state = (r, c)
            
            # Don't draw arrows on walls or the goal
            if maze_obj.maze[r][c] == 1 or current_state == maze_obj.end_position:
                continue
                
            x_center = 200 + c * size + size // 2
            y_center = 500 - r * size + size // 2
            
            if current_state in q_table:
                best_action = max(q_table[current_state], key=q_table[current_state].get)
                dx, dy = arrow_map[best_action]
                
                start_pos = (x_center - dx*15, y_center - dy*15)
                end_pos = (x_center + dx*15, y_center + dy*15)
                
                pygame.draw.line(screen, (0, 0, 0), start_pos, end_pos, 3)
                
                # Draw arrow head
                if dx == 1: # Right
                    pygame.draw.line(screen, (0, 0, 0), end_pos, (end_pos[0]-8, end_pos[1]-5), 3)
                    pygame.draw.line(screen, (0, 0, 0), end_pos, (end_pos[0]-8, end_pos[1]+5), 3)
                elif dx == -1: # Left
                    pygame.draw.line(screen, (0, 0, 0), end_pos, (end_pos[0]+8, end_pos[1]-5), 3)
                    pygame.draw.line(screen, (0, 0, 0), end_pos, (end_pos[0]+8, end_pos[1]+5), 3)
                elif dy == 1: # Down
                    pygame.draw.line(screen, (0, 0, 0), end_pos, (end_pos[0]-5, end_pos[1]-8), 3)
                    pygame.draw.line(screen, (0, 0, 0), end_pos, (end_pos[0]+5, end_pos[1]-8), 3)
                elif dy == -1: # Up
                    pygame.draw.line(screen, (0, 0, 0), end_pos, (end_pos[0]-5, end_pos[1]+8), 3)
                    pygame.draw.line(screen, (0, 0, 0), end_pos, (end_pos[0]+5, end_pos[1]+8), 3)
    
    pygame.display.flip()
    
    # Keep window open
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    pygame.quit()

# Run the training with visualization
if __name__ == "__main__":
    # Global maze layout for the drawing function
    maze_layout = [[0] * 5 for n in range(3)]
    maze_layout[0][2] = 1
    maze_layout[1][2] = 1
    
    q_table, trained_maze = train_with_full_visualization()
    q_table_2, trained_maze_2 = train_using_sarsa()

    print("Training completed! Showing final policy...")
    show_final_policy(q_table, trained_maze)
    plt.plot(trained_maze.rewards, color='red')
    plt.plot(trained_maze_2.rewards_sarsa, color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.show()