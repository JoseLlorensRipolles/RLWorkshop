import gym
import torch
import numpy as np
import random
from arquitecture import NeuralNetwork

EPISODES = 100
EPSILON = 0.1


def get_action(net, s):
    q = net(torch.from_numpy(s)).detach().numpy()
    if random.random() < EPSILON:
        return random.randint(0, len(q) - 1)
    return np.argmax(q)


if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    net = NeuralNetwork()
    for episode in range(EPISODES):
        s = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            s_prime, r, done, _ = env.step(get_action(net, s))
            total_reward += r
            s = s_prime
        print(f"Total reward for episode #{episode}: {total_reward}")
    env.close()