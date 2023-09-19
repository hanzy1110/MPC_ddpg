# import argparse
import logging
import os
import pathlib

# import gym
import numpy as np
import torch

# from wrappers import NormalizedActions
import time

from src.main_window import MainControlLoop, BATCH_SIZE
from src.env import AmeSimEnv, read_state_files

# Create logger
DATA_FILE = "logs/data.txt"
DATA_DIR = pathlib.Path(__file__).parent / "data"
LOG_FILE = pathlib.Path(__file__).parent / "log/events.log"
RANDOM_SEED = 123456
CHECKPOINT_DIR = pathlib.Path(__file__).parent / "checkpoints"

MAX_EPISODES = 10
MAX_TIMESTEPS = 100
MAX_ITERS = 100
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(
    filename=LOG_FILE, encoding="utf-8", level=logging.DEBUG, format=FORMAT
)

# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Some parameters, which are not saved in the network files
gamma = 0.99  # discount factor for reward (default: 0.99)
tau = 0.001  # discount factor for model (default: 0.001)
hidden_size = (
    10,
    20,
)  # size of the hidden layers (Deepmind: 400 and 300; OpenAI: 64)

if __name__ == "__main__":
    logging.info("Using device: {}".format(device))

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
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0
    time_last_checkpoint = time.time()

    while timestep <= MAX_TIMESTEPS:
        epoch_return = 0
        state = torch.Tensor([env.reset()]).to(device)
        # while True:

        controller.last_output_dis_target = read_state_files(timestep)[1]

        # AGENT (MainControlLoop) returns an ACTION
        # env returns enviroment response!

        it = 0
        while True:
            logging.info(
                f"Timestep {timestep} iter {it}"
            )
            if not isinstance(state, torch.Tensor):
                state = torch.Tensor([state]).to(device)

            actions = controller.control_step(timestep, state)

            # TODO convert actions to torch?
            # response = env.step(actions.cpu().numpy()[0], timestep)
            response = env.step(actions, timestep)
            timestep += 1
            epoch_return += response.reward

            controller.update_memory(response, actions)

            controller.update_references(response.next_state)
            state = response.next_state
            env.set_target(state)

            if len(controller.memories["agent_up"]) > BATCH_SIZE:
                controller.train_step(controller.ddpg_agent_up, "agent_up")
                controller.train_step(controller.ddpg_agent_down, "agent_down")
                controller.train_step(controller.ddpg_agent_in3, "agent_in3")

            it += 1
            if it >= MAX_ITERS:
                break
            # if done:
            #     break

        controller.epoch_data["rewards"].append(epoch_return)
        controller.epoch_data["epoch_value_loss"].append(controller.epoch_value_loss)
        controller.epoch_data["epoch_policy_loss"].append(controller.epoch_policy_loss)

        # # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
        # if timestep >= 10000 * t:
        #     t += 1
        #     test_rewards = []
        #     for _ in range(args.n_test_cycles):
        #         state = torch.Tensor([env.reset()]).to(device)
        #         test_reward = 0
        #         while True:
        #             if args.render_eval:
        #                 env.render()

        #             action = agent.calc_action(state)  # Selection without noise

        #             next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        #             test_reward += reward

        #             next_state = torch.Tensor([next_state]).to(device)

        #             state = next_state
        #             if done:
        #                 break
        #         test_rewards.append(test_reward)

        #     mean_test_rewards.append(np.mean(test_rewards))

        #     for name, param in agent.actor.named_parameters():
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        #     for name, param in agent.critic.named_parameters():
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        #     writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], epoch)
        #     logger.info("Epoch: {}, current timestep: {}, last reward: {}, "
        #                 "mean reward: {}, mean test reward {}".format(epoch,
        #                                                               timestep,
        #                                                               rewards[-1],
        #                                                               np.mean(rewards[-10:]),
        #                                                               np.mean(test_rewards)))

        #     # Save if the mean of the last three averaged rewards while testing
        #     # is greater than the specified reward threshold
        #     # TODO: Option if no reward threshold is given
        #     if np.mean(mean_test_rewards[-3:]) >= reward_threshold:
        #         agent.save_checkpoint(timestep, memory)
        #         time_last_checkpoint = time.time()
        #         logger.info('Saved model at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

        epoch += 1

    controller.save_checkpoint(timestep)
    # env.close()
