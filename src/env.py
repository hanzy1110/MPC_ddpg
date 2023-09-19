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

DATA_VELOCITY = pd.read_csv(DATA_DIR / "last_output_velocity_target.csv")
DATA_DIS = pd.read_csv(DATA_DIR / "last_output_dis_target.csv")

OUT_1 = pd.read_csv(DATA_DIR / "output1.csv")
OUT_2 = pd.read_csv(DATA_DIR / "output2.csv")
OUT_3 = pd.read_csv(DATA_DIR / "output3.csv")

MAX_EPISODES = 10

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(
    filename=LOG_FILE, encoding="utf-8", level=logging.DEBUG, format=FORMAT
)

SYSResponse = namedtuple("SYSResponse", ["state", "next_state","target", "reward", "done"])

def read_state_files(t):
    """Read and output enviroment data with the appropriate form"""
    velocity = DATA_VELOCITY["velocity"].values
    dis = DATA_DIS["signal"].values
    return np.array([velocity[t], dis[t]])

def read_output_files(t):
    out_1 = OUT_1["routput"].values[t]
    # out_2 = OUT_2["routput"].values[t]
    # out_3 = OUT_3["routput"].values[t]
    # return np.array([out_1, out_2, out_3])
    return np.array([out_1, 0,0,0,0,0])

class AmeSimEnv:
    def __init__(self) -> None:
        pass

    def reset(self):
        return read_output_files(0)

    def set_target(self, target):
        self.target = target

    def step(self, action, time):
        logging.info(f"Action: {action}")
        target = read_state_files(time)
        next_target = read_state_files(time + 1)

        state = read_output_files(time)
        next_state = read_output_files(time + 1)


        done = False
        reward = np.linalg.norm(target[0]-state[0])

        return SYSResponse(state, next_state, next_target,reward, done)
