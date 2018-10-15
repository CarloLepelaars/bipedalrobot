# Random numbers
import random
# Matrix math
import numpy as np
# For storing weights
import pickle
# Evolution Strategy (ES) algorithm.
# Check out https://blog.openai.com/evolution-strategies for more information
from evostra import EvolutionStrategy
# OpenAI Gym environment
import gym

class Agent:
    agent_hist = 1
    population = 50
    eps_avg = 1
    sigma = 0.2
    lr = 0.1 # Learning Rate
    init_explore = 0.9
    final_explore = 0.1
    explore_steps = 1E+6 

    def __init__(self):
    	# Initializes environment, Model, Algorithm and Exploration
        self.env = gym.make('BipedalWalker-v2')
        self.model = Model()
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.population, self.sigma, self.lr)
        self.exploration = self.init_explore

    def get_predicted_action(self, sequence):
    	# Retreive the predicted action
        prediction = self.model.predict(np.array(sequence))
        return prediction

    def load(self, filename='weights.pkl'):
		# Loads weights for agent_play
        with open(filename,'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()

    def save(self, filename='weights.pkl'):
    	# Saves weigths to Pickle file
        with open(filename, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)

    def play(self, episodes, render=True):
        # Run the model in the OpenAI environment
        self.model.set_weights(self.es.weights)
        for episode in range(episodes):
            total_reward = 0
            observation = self.env.reset()
            sequence = [observation]*self.agent_hist
            done = False
            while not done:
                if render:
                    self.env.render()
                action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)
            print("Total reward:", total_reward)

    def train(self, iterations):
        # Begin training
        self.es.run(iterations, print_step=1)

    def get_reward(self, weights):
        # Initialize reward
        total_reward = 0.0
        self.model.set_weights(weights)

        # Calculate reward
        for episode in range(self.eps_avg):
            observation = self.env.reset()
            sequence = [observation]*self.agent_hist
            done = False
            while not done:
                self.exploration = max(self.final_explore, self.exploration - self.init_explore/self.explore_steps)
                if random.random() < self.exploration:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)
        return total_reward/self.eps_avg

class Model(object):

    def __init__(self):
        # Initialize weights using zero matrices
        self.weights = [np.zeros(shape=(24, 16)), np.zeros(shape=(16, 16)), np.zeros(shape=(16, 4))]

    def predict(self, inp):
    	# Predict action
        out = np.expand_dims(inp.flatten(), 0)
        out = out / np.linalg.norm(out)
        for layer in self.weights:
            out = np.dot(out, layer)
        return out[0]

    def get_weights(self):
    	# Retreive weigths
        return self.weights

    def set_weights(self, weights):
    	# Set new weights
        self.weights = weights
