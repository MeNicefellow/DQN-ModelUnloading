# Model Unloading Optimization using DQN

## Project Overview

This project implements a reinforcement learning (RL) solution using Deep Q-Networks (DQN) to optimize the decision of unloading a language model on a server. The goal is to minimize resource usage while ensuring the model is available for user requests, thereby balancing performance and cost.

## Features

- **Simulation of User Behavior**: Mimics user interactions with a server to test different strategies for model loading and unloading.
- **Deep Q-Networks (DQN)**: Utilizes a DQN with separate policy and target networks to stabilize the learning process.
- **Performance Tracking**: Tracks and prints the reward improvement over episodes to evaluate the learning progress.

## Dependencies

To run this project, you will need the following:
- Python 3.x
- PyTorch
- NumPy

You can install the necessary libraries using pip:

```bash
pip install torch numpy
```

## Setup

1. Clone the repository to your local machine.
2. Ensure that all required dependencies are installed.
3. Navigate to the project directory.

## Running the Code

To start the training process, run the following command in your terminal:

```bash
python main.py
```

## Output

The script will output the average rewards every 10 episodes to the console, allowing you to monitor the improvement in decision-making ability of the agent over time.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your suggested changes.

## License

This project is open-source and available under the APACHE 2.0 License.

## Discord Server

Join our Discord server [here](https://discord.gg/xhcBDEM3).


## Feeling Generous? 😊
Eager to buy me a cup of 2$ coffe or iced tea?🍵☕ Sure, here is the link: [https://ko-fi.com/drnicefellow](https://ko-fi.com/drnicefellow). Please add a note on which one you want me to drink?
