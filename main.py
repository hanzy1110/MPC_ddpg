# import argparse
import logging
import os
import pathlib
import json

# import gym
import numpy as np
import torch

import pandas as pd

# from wrappers import NormalizedActions
import time
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

from src.main_window import MainControlLoop, BATCH_SIZE, DDPGRes
from src.env import AmeSimEnv, read_state_files, SYSResponse

# Create logger

# PATH = os.getcwd()
PATH = __file__
DATA_DIR = pathlib.Path(PATH).parent / "data"
PLOTS_DIR = pathlib.Path(PATH).parent / "plots"
LOG_FILE = pathlib.Path(PATH).parent / "log/events.log"
RANDOM_SEED = 123456
CHECKPOINT_DIR = pathlib.Path(PATH).parent / "checkpoints"

MAX_EPISODES = 10
MAX_TIMESTEPS = 1300
MAX_ITERS = 50
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format=FORMAT)


# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (torch.Tensor,)):
            return obj.detach().numpy().tolist()

        return json.JSONEncoder.default(self, obj)


# Some parameters, which are not saved in the network files
gamma = 0.99  # discount factor for reward (default: 0.99)
tau = 0.001  # discount factor for model (default: 0.001)
hidden_size = (
    10,
    20,
)  # size of the hidden layers (Deepmind: 400 and 300; OpenAI: 64)

if __name__ == "__main__":
    logging.info("Using device: {}".format(device))
    logging.info("Previous plots")

    fig, ax = plt.subplots(2, 1, sharex=True)
    data_disp = pd.read_csv(DATA_DIR / "last_output_dis_target.csv")
    data_vel = pd.read_csv(DATA_DIR / "last_output_velocity_target.csv")
    ax[0].plot(data_disp["time"], data_disp["signal"])
    ax[1].plot(data_vel["time"], data_vel["velocity"])

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Displacement")
    ax[1].set_ylabel("Velocity")

    plt.savefig(PLOTS_DIR / "disp_target.png")

    # Create the env
    # kwargs = dict()
    # TODO Maybe usefull
    # env = NormalizedActions(env)

    # Setting rnd seed for reproducibility
    # env.seed(args.seed)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ddpg_params = {
        "num_inputs": 6,
        "gamma": gamma,
        "tau": tau,
        "hidden_size": hidden_size,
        "checkpoint_dir": CHECKPOINT_DIR,
    }

    controller = MainControlLoop(ddpg_params)
    env = AmeSimEnv()

    # Initialize OU-Noise
    # This is on a per-agent basis
    timestep = 1
    # rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    states = []
    epoch = 0
    t = 0
    time_last_checkpoint = time.time()
    responses = []
    action_arr = []

    while timestep <= MAX_TIMESTEPS:
        if env.is_running:
            epoch_return = 0
            state = torch.Tensor([env.reset()]).to(device)
            # while True:

            controller.last_output_dis_target = read_state_files(timestep)[1]

            # AGENT (MainControlLoop) returns an ACTION
            # env returns enviroment response!

            it = 0
            while True:
                logging.info("--x--" * 20)
                logging.info(f"Timestep {timestep} iter {it}")
                if not isinstance(state, torch.Tensor):
                    state = torch.Tensor([state]).to(device)

                actions = controller.control_step(timestep, state)

                # TODO convert actions to torch?
                # response = env.step(actions.cpu().numpy()[0], timestep)
                response = env.step(actions, timestep)
                epoch_return += response.reward

                responses.append({"response": response, "actions": actions})

                controller.update_memory(response, actions)

                states.append(response.next_state)
                controller.update_references(response.next_state)
                state = response.next_state
                env.set_target(state)

                if len(controller.memories["agent_up"]) > BATCH_SIZE:
                    logging.info(f"TRAINING STEP => {controller.memories}")
                    controller.train_step(controller.ddpg_agent_up, "agent_up")
                    controller.train_step(controller.ddpg_agent_down, "agent_down")
                    controller.train_step(controller.ddpg_agent_in3, "agent_in3")

                it += 1
                if it >= MAX_ITERS:
                    break
                # if done:
                #     break

            timestep += 1
            controller.epoch_data["rewards"].append(epoch_return)

            controller.epoch_data["epoch_value_loss"].append(
                controller.epoch_value_loss
            )
            controller.epoch_data["epoch_policy_loss"].append(
                controller.epoch_policy_loss
            )

    with open(DATA_DIR / "results.json", "w") as _file:
        # for r in responses:
        json.dump(responses, _file, indent=4, cls=NumpyEncoder)

    def flatten_dict(LD):
        out = {}
        for vals in LD:
            for k, v in vals.items():
                try:
                    out[k].append(v)
                except Exception as e:
                    out[k] = []
                    out[k].append(v)

        return out

    fig, ax = plt.subplots(2, 2)

    fig.set_size_inches(6.2, 6.2)

    ax[0, 0].plot(controller.epoch_data["rewards"])
    ax[0, 0].set_title("Rewards")

    for k, v in flatten_dict(controller.epoch_data["epoch_value_loss"]).items():
        ax[0, 1].plot(v, label=k)
    for k, v in flatten_dict(controller.epoch_data["epoch_policy_loss"]).items():
        ax[1, 0].plot(v, label=k)

    ax[1, 1].plot([s[1] for s in states], label="Displacement")

    ax[0, 1].set_title("Epoch Value Loss")
    ax[1, 0].set_title("Epoch Policy Loss")
    ax[1, 1].set_title("Displacement")

    for a in ax.flatten():
        a.legend()

    plt.savefig(PLOTS_DIR / "training_plot.jpg", dpi=600)
    controller.save_checkpoint(timestep)

    env.close()
