import time
import pathlib
import numpy as np
import pandas as pd
import gymnasium as gym
import torch
from gymnasium.spaces import Box

from collections import namedtuple

# import tkinter
# import AmeCommunication

import logging
from .mpc import MPCControllerdown, MPCControllerup, MPCControllerupp, MPCResult
from .helpers import Chart, Chrono
from .ddpg import DDPG
from .drl_utils.noise import OrnsteinUhlenbeckActionNoise
from .drl_utils.replay_memory import ReplayMemory, Transition

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_FILE = "logs/data.txt"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
LOG_FILE = pathlib.Path(__file__).parent.parent / "log/events.log"

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format=FORMAT)
# DATA_VELOCITY = pd.read_csv(DATA_DIR / "last_output_velocity_target.data")
# DATA_DIS = pd.read_csv(DATA_DIR / "last_output_dis_target.data")

DDPGRes = namedtuple("DDPGRes", ["action", "q_value"])
ENVRes = namedtuple("ENVRes", ["current_state", "reward"])
SYSResponse = namedtuple("SYSResponse", ["velocity", "displacement"])

MEMORY_SIZE = 10
BATCH_SIZE = 10

def parse_mpc_response(resp:MPCResult):
    res = np.hstack([resp.control, resp.state])
    return torch.Tensor(res).to(DEVICE)


class MainControlLoop(object):
    def __init__(self, ddpg_params):
        self.is_running = False
        self.rett = None

        self.chrono = Chrono()
        self.last_refresh_time = 0

        self.epoch_data = {
            "rewards": [],
            "epoch_value_loss": [],
            "epoch_policy_loss": [],
        }

        # MPC Control:
        self.mpc_controllerup = MPCControllerup()
        self.mpc_controllerdown = MPCControllerdown()
        self.mpc_controllerupp = MPCControllerupp()

        self.action_spaces = {
            "agent_up": Box(-10, 0, dtype=np.float32),
            "agent_down": Box(0, 1, dtype=np.float32),
            "agent_in3": Box(0, 250, dtype=np.float32),
        }

        self.memories = {
            k: ReplayMemory(MEMORY_SIZE) for k in self.action_spaces.keys()
        }
        self.epoch_value_loss = {k: 0 for k in self.action_spaces.keys()}
        self.epoch_policy_loss = {k: 0 for k in self.action_spaces.keys()}

        # self.ddpg_agent_up = DDPG(0.1, 0.1, (10, 10), 3, controlup_state_space)
        self.ddpg_agent_up = self.set_agent(
            self.action_spaces["agent_up"], "agent_up", **ddpg_params
        )
        self.ddpg_agent_down = self.set_agent(
            self.action_spaces["agent_down"], "agent_down", **ddpg_params
        )
        self.ddpg_agent_in3 = self.set_agent(
            self.action_spaces["agent_in3"], "agent_in3", **ddpg_params
        )

    def set_target(self, target_disp):
        self.last_output_dis_target = target_disp
        # self.last_output_vel_target = target_disp

    def set_agent(self, action_space, name, **params):
        logging.info(f"Setting Agent: {name}")
        chk_dir = params.pop("checkpoint_dir")

        checkpoint_dir = chk_dir / name

        agent = DDPG(
            name=name,
            action_space=action_space,
            checkpoint_dir=checkpoint_dir,
            **params,
        )

        try:
            agent.load_checkpoint()
        except ValueError as e:
            logging.warning(f"No Checkpoint found! {e}")
        agent.set_eval()
        return agent

    def get_noise(self, name):
        # TODO add standard deviation to noise
        nb_actions = self.action_spaces[name].shape[-1]
        ou_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(nb_actions), sigma=float(1) * np.ones(nb_actions)
        )
        return ou_noise

    def compute_actions(self, agent, state, name):
        action = agent.calc_action(state, action_noise=self.get_noise(name))
        logging.info(f"ACTION FROM DDPG {name} => {action}")
        print(f"ACTION FROM DDPG {name} => {action}")
        print(f"STATE {name} => {state}")
        q_value = agent.critic(state, action)
        return DDPGRes(action=action, q_value=q_value)

    def update_references(self, current_state):
        self.last_output_velocity = current_state[0]
        self.last_output_dis_target = current_state[1]

    def update_memory(self, response, actions):
        mask = torch.Tensor([response.done]).to(DEVICE)
        reward = torch.Tensor([response.reward]).to(DEVICE)
        next_state = torch.Tensor([response.next_state]).to(DEVICE)
        state = torch.Tensor([response.state]).to(DEVICE)

        for agent_name in self.action_spaces.keys():
            self.memories[agent_name].push(
                state, actions[agent_name], mask, next_state, reward
            )

    def train_step(self, agent, agent_name):
        transitions = self.memories[agent_name].sample(BATCH_SIZE)
        # Transpose the batch
        # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
        batch = Transition(*zip(*transitions))

        # Update actor and critic according to the batch
        value_loss, policy_loss = agent.update_params(batch)
        self.epoch_value_loss[agent_name] += value_loss
        self.epoch_policy_loss[agent_name] += policy_loss

    def control_step(self, t, current_state):
        # while self.is_running:
        # t = self.chrono.get_time()
        # Modify the state accordingly-self.mpc_controllerupp.controlup(current_state, self.last_output_dis_target)[1]*20
        current_state = current_state.numpy().reshape((6,))
        logging.info(f"CURRENT STATE -> {current_state}")
        state_val_in3 = (
            self.mpc_controllerupp.controlup(current_state, self.last_output_dis_target)
            # * 23
        )
        state_val_up = (
            self.mpc_controllerup.controlup(current_state, self.last_output_dis_target)
            # * 1.55
        )
        state_val_down = (
            self.mpc_controllerdown.controldown(
                current_state, self.last_output_dis_target
            )
            # * 0.52
        )

        pred_state_val_in3 = parse_mpc_response(state_val_in3)
        pred_state_val_up = parse_mpc_response(state_val_up)
        pred_state_val_down = parse_mpc_response(state_val_down)
        # pred_state_val_in3 = torch.Tensor([state_val_in3.state, state_val_in3.control]).to(DEVICE)
        # pred_state_val_up = torch.Tensor([state_val_up.state, state_val_up.control]).to(DEVICE)
        # pred_state_val_down = torch.Tensor([state_val_down.state, state_val_down.control]).to(DEVICE)

        # predicted state ==> a function of the the MPC control
        output_ddpg_up = self.compute_actions(
            self.ddpg_agent_up, pred_state_val_up, name="agent_up"
        )
        output_ddpg_down = self.compute_actions(
            self.ddpg_agent_down, pred_state_val_down, name="agent_down"
        )
        output_ddpg_in3 = self.compute_actions(
            self.ddpg_agent_in3, pred_state_val_in3, name="agent_in3"
        )

        actions = {
            "agent_up": output_ddpg_up,
            "agent_down": output_ddpg_down,
            "agent_in3": output_ddpg_in3,
        }

        logging.debug(f"CONTROL UP : {output_ddpg_up}")
        logging.debug(f"CONTROL DOWN : {output_ddpg_down}")
        logging.debug(f"CONTROL IN3 : {output_ddpg_in3}")

        # Output actions should go to the enviroment
        # but we just read the outcome of a previous simulation

        # Just Communication with Amesim?
        # Here it returns the output from the rectangle
        # The data files should be read here
        # rett = self.shm.exchange(
        #     [0.0, t, output_val_in3, output_val_down, output_val_up]
        # )

        return actions

    def save_checkpoint(self, timestep):
        self.ddpg_agent_up.save_checkpoint(timestep, self.memories["agent_up"])
        self.ddpg_agent_down.save_checkpoint(timestep, self.memories["agent_down"])
        self.ddpg_agent_in3.save_checkpoint(timestep, self.memories["agent_in3"])

        logging.info(
            "Saved model at endtime {}".format(
                time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.localtime())
            )
        )
        logging.info(
            "Stopping training at {}".format(
                time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.localtime())
            )
        )
