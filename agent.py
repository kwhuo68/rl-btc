import numpy as np
import math
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import History 


class Agent:
	def __init__(self, feature_size, window, action_size, experience_replay, environment):
		self.feature_size = feature_size
		self.window = window
		self.action_size = action_size
		self.model = self.build_model()
		self.epsilon = 0.5
		self.min_epsilon = 0.1 
		self.epsilon_decay = 0.95
		self.exp_replay = experience_replay
		self.env = environment	
		self.state_history = np.zeros([window, feature_size], dtype = np.float32)
		self.buffer_size = 40
		self.batch_size = 20

	class LossHistory(keras.callbacks.Callback):
		def on_train_begin(self, logs={}):
			self.losses = []
		def on_batch_end(self, batch, logs={}):
			self.losses.append(logs.get('loss'))

	def build_model(self):
		model = Sequential()
		model.add(Dense(units = 64, input_shape = (6, ), activation = 'relu'))
		model.add(Dense(units = 32, activation = 'relu'))
		model.add(Dense(units = 8, activation = 'relu'))
		model.add(Dense(self.action_size, activation = 'linear'))
		model.compile(loss = 'mse', optimizer = Adam(lr = 0.001))
		return model


	def train(self):
		num_episodes, episode_reward, total_reward = 25, 0, 0
		iterations = 0
		episode_total_loss = []
		episode_total_reward = []
		(self.state_history, self.exp_replay) = self.env.reset(self.state_history, self.exp_replay)

		for i in range(num_episodes):
			episode_loss = 0
			episode_reward = 0

			#Starting state and previous state
			input_state = np.array([self.state_history[-1].T])
			prev_state = input_state
			termination = False

			while not termination:
				iterations += 1
				prev_state = input_state
				#Take random action, else model picks
				if(np.random.rand() < self.epsilon):
					action = random.choice(list(self.env.actions.keys()))
					isRandom = True
				else:
					q = self.model.predict(input_state, batch_size = 1)
					action = np.argmax(q[0])

				#Carry out action and observe reward and new state s', store experience in replay
				input_state, reward, termination = self.env.step(action)
				episode_reward += reward 
				self.exp_replay.store([prev_state, action, reward, input_state], termination)

				#Sample random transitions from replay memory and calculate target for each minibatch transition 
				if(self.exp_replay.size() > self.buffer_size):
					inputs, targets = self.exp_replay.sample_batch(self.model, batch_size = self.window)
					hist = History()
					self.model.fit(inputs, targets, batch_size = self.window, nb_epoch = 1, verbose = 0, callbacks = [hist])
					episode_loss += hist.history['loss'][0]

			self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
			print("Episode {} , with loss of {}, and reward of {}".format(i, episode_loss, episode_reward))

			if termination:
				(self.state_history, self.exp_replay) = self.env.reset(self.state_history, self.exp_replay)
				i += 1
				episode_total_loss.append(episode_loss)
				episode_total_reward.append(episode_reward)
		
		print("Episode total loss: " + str(episode_total_loss))
		print("Episode total reward: " + str(episode_total_reward))
		return 0


	def test(self, test_history):
		step_total_reward = []
		total_reward = 0

		(self.state_history, self.exp_replay) = self.env.reset(self.state_history, self.exp_replay, test = True)
		input_state = np.array([test_history[-1].T])
		#Iterate over testing data
		for i in range(len(test_history)):
			q = self.model.predict(input_state, batch_size = 1)
			action = np.argmax(q[0])
			input_state, reward, termination = self.env.step(action, test=True)
			step_total_reward.append(reward)
			total_reward += reward

		print("Final reward: " + str(step_total_reward[-1]))
		return 0


