import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
import numpy as np
import argparse
import wandb
import torch

def evaluate(
    model,
    num_episodes = 10,
    deterministic = True,
) -> float:
    """
    Evaluate an RL agent for `num_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param num_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward for the last `num_episodes`
    """
    # This function will only work for a single environment
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episode_rewards = []
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _info = vec_env.step(action)
            episode_rewards.append(reward)
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} - Num episodes: {num_episodes}")

    return mean_episode_reward


def train(args):
    total_steps = args.env_steps
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = make_vec_env(args.env_name, n_envs=1)
    if args.algo == 'ppo':
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1)
    elif args.algo == 'a2c':
        from stable_baselines3 import A2C
        model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=1e-5)
    epoch_length = 10000

    wandb.init(entity="a2i", project="polygrad_results", group=args.group)
    for i in range(total_steps // epoch_length):
        model.learn(total_timesteps=epoch_length)
        mean_reward = evaluate(model, num_episodes=10, deterministic=True)
        print(f"Mean reward at step {i}: {mean_reward:.2f}")
        wandb.log({'mean_reward': mean_reward}, step=(i + 1) * epoch_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--group', default='default')
    parser.add_argument(
        '--env_steps', type=int, default=1000000)
    parser.add_argument(
        '--seed', type=int, default=0)
    parser.add_argument(
        '--algo', default='ppo')
    parser.add_argument(
        '--env_name', default='Hopper-v3')
    args = parser.parse_args()
    train(args)