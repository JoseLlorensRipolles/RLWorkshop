import gym
import torch
import numpy as np

EPISODES = 100

if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    for episode in range(EPISODES):
        env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            _, r, done, _ = env.step(env.action_space.sample())
            total_reward += r
        print(f"Total reward for episode #{episode}: {total_reward}")
    env.close()