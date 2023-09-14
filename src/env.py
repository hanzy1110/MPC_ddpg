import logging
import os
import pathlib
import pandas as pd
import numpy as np
from collections import namedtuple

DATA_FILE = "logs/data.txt"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
LOG_FILE = pathlib.Path(__file__).parent.parent / "log/events.log"
RANDOM_SEED = 123456
CHECKPOINT_DIR = pathlib.Path(__file__).parent / "checkpoints"

DATA_VELOCITY = pd.read_csv(DATA_DIR / "last_output_velocity_target.data")
DATA_DIS = pd.read_csv(DATA_DIR / "last_output_dis_target.data")

MAX_EPISODES = 10

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(
    filename=LOG_FILE, encoding="utf-8", level=logging.DEBUG, format=FORMAT
)

SYSResponse = namedtuple("SYSResponse", ["state", "next_state", "reward", "done"])

def read_output_files(t):
    """Read and output enviroment data with the appropriate form"""
    velocity = DATA_VELOCITY["velocity"].values
    dis = DATA_DIS["signal"].values
    return velocity[t], dis[t]

class AmeSimEnv:
    def __init__(self) -> None:
        pass

    def reset(self):
        velocity, disp = read_output_files(0)
        return np.array([velocity, disp])

    def set_target(self, target):
        self.target = target

    def step(self, action, time):
        logging.info(f"Action: {action}")
        velocity, disp = read_output_files(time)
        state = np.array([velocity, disp])
        next_velocity, next_disp = read_output_files(time + 1)
        next_state = np.array([next_velocity, next_disp])

        done = False
        reward = np.linalg.norm(np.array([next_velocity, next_disp])-self.target)

        return SYSResponse(state, next_state, reward, done)
