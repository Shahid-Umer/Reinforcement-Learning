""" The A3C algorithm is a powerful deep reinforcement learning technique used for solving complex decision-making
problems. In this project, I implemented the A3C algorithm for the CartPole-v1 environment using the PyTorch library.
PyTorch provides an easy-to-use and efficient platform for building deep learning models, and A3C is a natural fit
for PyTorch's dynamic computation graph.

By using this code of A3C for CartPole-v1, you will gain a deep understanding of how reinforcement learning works and
how to leverage PyTorch's capabilities to train powerful neural networks for complex tasks. """
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
import os

# Hyperparameters
LR = 0.0001
GAMMA = 0.99
TAU = 1.00
ENTROPY_BETA = 0.001
EPISODES = 1000
TIMESTEPS = 200
N_WORKERS = 4
N_STEPS = 20

# Environment
env = gym.make('CartPole-v1')
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.n


# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(S_DIM, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, A_DIM)
        self.critic = nn.Linear(128, 1)
        self.std = nn.Parameter(torch.ones(1, A_DIM))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.actor(x)
        dist = Categorical(F.softmax(mean, dim=-1))
        value = self.critic(x)
        return dist, value


# Shared global network
global_net = ActorCritic()
global_net.share_memory()

# Optimizer
optimizer = optim.Adam(global_net.parameters(), lr=LR)


# Parallel worker
class Worker(mp.Process):
    def __init__(self, idx, global_net, optimizer, queue):
        super(Worker, self).__init__()
        self.worker_idx = idx
        self.local_net = ActorCritic()
        self.local_net.load_state_dict(global_net.state_dict())
        self.optimizer = optimizer
        self.queue = queue

    def run(self):
        env = gym.make('CartPole-v1')
        for episode in range(EPISODES):
            done = False
            state, info = env.reset()
            episode_reward = 0
            log_probs = []
            values = []
            rewards = []

            for t in range(TIMESTEPS):
                state = torch.FloatTensor(state)
                dist, value = self.local_net(state)
                action = dist.sample()
                next_state, reward, done, _, _ = env.step(action.item())
                episode_reward += reward
                log_prob = dist.log_prob(action)
                entropy = dist.entropy().mean()
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)

                if done:
                    self.queue.put((episode_reward, None, None))
                    break

                state = next_state

            if not done:
                _, value = self.local_net(torch.FloatTensor(next_state))
                returns = []
                discounted_reward = value.detach()
                for reward in reversed(rewards):
                    discounted_reward = reward + GAMMA * discounted_reward
                    returns.insert(0, discounted_reward)
                returns = torch.tensor(returns)
                log_probs = torch.stack(log_probs)
                values = torch.stack(values)
                advantage = returns - values
                actor_loss = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()
                entropy_loss = -ENTROPY_BETA * entropy
                loss = actor_loss + 0.5 * critic_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), TAU)
                self.optimizer.step()

            # Update local network
            self.local_net.load_state_dict(global_net.state_dict())

            if episode % 50 == 0:
                print(f"Worker {self.worker_idx}, Episode {episode}, Reward: {episode_reward:.2f}")


def save(model, optimizer):

    torch.save({'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, 'last_brain.pth')


def load(model, optimizer):
    if os.path.isfile('last_brain.pth'):
        print("=> loading checkpoint... ")
        checkpoint = torch.load('last_brain.pth')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("done !")
        return model, optimizer

    else:
        print("no checkpoint found...")


# Main process
if __name__ == '__main__':
    queue = torch.multiprocessing.Queue()
    # Shared global network
    global_net = ActorCritic()
    global_net.share_memory()
    # Optimizer
    optimizer = optim.Adam(global_net.parameters(), lr=LR)

    global_net, optimizer = load(global_net, optimizer)

    workers = [Worker(idx, global_net, optimizer, queue) for idx in range(N_WORKERS)]
    [w.start() for w in workers]
    episode_rewards = []

    for episode in range(EPISODES):
        rewards = []
        for worker in workers:
            episode_reward, _, _ = queue.get()
            rewards.append(episode_reward)
        episode_reward = sum(rewards) / len(rewards)
        episode_rewards.append(episode_reward)

        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")

    [w.join() for w in workers]

    save(global_net, optimizer)

    # Plot rewards
    import matplotlib.pyplot as plt

    plt.plot(episode_rewards)
    plt.title('Episode rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
