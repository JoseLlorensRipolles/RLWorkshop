from collections import deque
import gym
import torch
import numpy as np
import random

import torch.nn.functional as F
from arquitecture import NeuralNetwork
from torch import optim

EPISODES = 10_000
EPSILON = 0.1
REPLAY_SIZE = 1000
GAMMA = 0.95


def get_action(net, s):
    q = net(torch.from_numpy(s)).detach().numpy()
    if random.random() < EPSILON:
        return random.randint(0, len(q) - 1)
    return np.argmax(q)


def train_net(net, optimizer, replay_buffer):
    tr_set = np.array(list(replay_buffer), dtype=object)
    tr_set = tr_set[np.random.randint(len(tr_set), size=int(REPLAY_SIZE / 2))]

    X = np.stack(tr_set[:, 0])
    y = net(torch.from_numpy(X)).detach().numpy()
    a = tr_set[:, 1].astype(int)
    r = tr_set[:, 2].astype(int)
    s_primes = np.stack(tr_set[:, 3])
    q_primes = net(torch.from_numpy(s_primes)).detach().numpy()
    max_q_primes = np.max(q_primes, axis=1)
    new_targets = r + max_q_primes * GAMMA
    done = tr_set[:, 4].astype(bool)

    non_done_idx = (~done).nonzero()
    y[non_done_idx, a[non_done_idx]] = new_targets[non_done_idx]
    done_idx = done.nonzero()
    y[done_idx, a[done_idx]] = -1

    data, target = torch.from_numpy(X), torch.from_numpy(y)
    optimizer.zero_grad()
    output = net(data)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    net = NeuralNetwork()
    optimizer = optim.Adam(net.parameters())
    replay_buffer = deque(maxlen=REPLAY_SIZE)
    for episode in range(EPISODES):
        s = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            a = get_action(net, s)
            s_prime, r, done, _ = env.step(a)
            total_reward += r
            replay_buffer.append([s, a, r, s_prime, done])
            if len(replay_buffer) == REPLAY_SIZE:
                train_net(net, optimizer, replay_buffer)
            s = s_prime
        print(f"Total reward for episode #{episode}: {total_reward}")
    env.close()