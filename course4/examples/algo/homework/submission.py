# # This is homework.
# # Load your model and submit this to Jidi

import torch
import os
import numpy as np

# load critic
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from critic import Critic


# TODO
class IQL:
    def __init__(self, state_dim, action_dim, hidden_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_size)

    def choose_action(self, observation):
        return torch.argmax(self.critic(torch.tensor(np.array(observation), dtype=torch.float)))

    def load(self, path):
        self.critic.load_state_dict(torch.load(path))


# TODO
def action_from_algo_to_env(joint_action):
    joint_action_ = []
    for a in range(n_player):
        action_a = joint_action
        each = [0] * action_dim
        each[action_a] = 1
        joint_action_.append(each)
    return joint_action_


# todo

n_player = 2
state_dim = 18
action_dim = 4
hidden_size = 64

# Once start to train, u can get saved model. Here we just say it is critic.pth.
critic_net = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'critic_1_10000.pth'
agent = IQL(state_dim, action_dim, hidden_size)
agent.load(critic_net)


def build_current_observation(observation):
    pass


# todo
def my_controller(observation, action_space, is_act_continuous=False):
    print(observation)
    next_state = get_observations_sw(observation, 0, obs_dim=state_dim)
    action = agent.choose_action(next_state)
    return action_from_algo_to_env(action)


def get_observations_sw(state, id, obs_dim):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map_sw(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    observations = np.zeros((1, obs_dim))  # todo
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    agents_index = [id]
    for i, element in enumerate(agents_index):
        # # self head position
        observations[i][:2] = snakes_positions_list[element][0][:]

        # head surroundings
        head_x = snakes_positions_list[element][0][1]
        head_y = snakes_positions_list[element][0][0]

        head_surrounding = get_surrounding_sw(state, board_width, board_height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions # todo: to check
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, element, 0)
        observations[i][16:] = snake_heads.flatten()[:]
    return observations.squeeze().tolist()


def make_grid_map_sw(board_width, board_height, beans_positions: list, snakes_positions: dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


def get_surrounding_sw(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding
