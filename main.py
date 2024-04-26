import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Parameters
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
BATCH_SIZE = 16
LR = 0.001
NUM_EPISODES = 500
PRINT_EVERY = 10


# Environment simulation
class ServerEnvironment:
    def __init__(self):
        self.model_loaded = True
        self.time_since_last_query = 0
        self.num_visits = 0
        self.num_queries = 0

    def get_state(self):
        return np.array([self.model_loaded, self.time_since_last_query, self.num_visits, self.num_queries])

    def step(self, action):
        reward = 0
        if action == 0:  # Unload model
            if self.model_loaded:
                self.model_loaded = False
                reward = -1  # Cost for unloading
        elif action == 1:  # Keep model loaded
            if not self.model_loaded:
                self.model_loaded = True
                reward = -1  # Cost for loading

        # Simulate user behavior
        if random.random() < 0.5:  # 50% chance of new user visit
            self.num_visits += 1
            num_user_queries = random.randint(0, 5)
            if num_user_queries > 0:
                self.num_queries += num_user_queries
                if self.model_loaded:
                    reward += num_user_queries * 0.5  # Reward for serving requests
                else:
                    reward -= num_user_queries * 0.5  # Penalty for not serving requests

        self.time_since_last_query += 1
        if self.num_queries > 0:
            self.num_queries -= 1
            self.time_since_last_query = 0

        return self.get_state(), reward, False  # No terminal state in this simulation

    def reset(self):
        self.model_loaded = True
        self.time_since_last_query = 0
        self.num_visits = 0
        self.num_queries = 0
        return self.get_state()


# Neural Network for Q-learning
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # State size to hidden layer
        self.fc2 = nn.Linear(16, 2)  # Output layer size (2 actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training loop with evaluation
def train_model():
    env = ServerEnvironment()
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = []

    steps_done = 0
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        for t in range(1000):  # Limit the number of steps per episode
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

            # Select and perform an action
            sample = random.random()
            if sample > eps_threshold:
                with torch.no_grad():
                    state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
                    action = policy_net(state_tensor).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(2)]], dtype=torch.long)
            action = action.cpu() # Move tensor back to CPU and get the value as a Python integer


            next_state, reward, done = env.step(action.item())
            total_reward += reward

            # Store the transition in memory
            memory.append((state, action.item(), next_state, reward))

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if len(memory) > BATCH_SIZE:
                transitions = random.sample(memory, BATCH_SIZE)
                batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
                batch_state = torch.tensor(batch_state, dtype=torch.float32).to(device)
                batch_action = torch.tensor(batch_action, dtype=torch.long).to(device)
                batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32).to(device)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)

                state_action_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1))
                next_state_values = target_net(batch_next_state).max(1)[0].detach()
                expected_state_action_values = (next_state_values * GAMMA) + batch_reward

                # Compute Huber loss
                loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_rewards.append(total_reward)
        if (episode + 1) % PRINT_EVERY == 0 or episode == 0:
            print(f'Episode {episode + 1}, Average Reward: {np.mean(episode_rewards[-PRINT_EVERY:])}')


train_model()
