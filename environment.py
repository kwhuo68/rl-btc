import numpy as np
import math
import random

class Environment:
	def __init__(self, horizon, train_price_history, test_price_history, history_length):
		self.actions = {2: 'LONG', 1: 'SHORT', 0: 'HOLD'}
		self.unit = 1e-2
		self.horizon = horizon
		self.train_price_history = train_price_history
		self.test_price_history = test_price_history
		self.history_length = history_length
		

	def reset(self, state_history, experience_replay, test = False):
		self.timesteps = 0
		self.long, self.short, self.bought, self.sold = 0.0, 0.0, 0.0, 0.0

		#Pick starting point randomly if testing
		if test == False:
			self.current_index = random.randint(self.history_length, len(self.train_price_history) - self.horizon)
		else:
			self.current_index = len(self.test_price_history)

		for state in self.train_price_history[self.current_index - self.history_length:self.current_index]:
			state_history[:-1] = state_history[1:]
			state_history[-1] = state 
			return (state_history, experience_replay)

	def step(self, action, test = False):
		curr_state = self.train_price_history[self.current_index] 
		curr_price = curr_state[0]

		#Long
		if action == 2: 
			self.long += 1
			self.bought += (self.unit * curr_price)
		
		#Short
		if action == 1:
			self.short += 1
			self.sold += (self.unit * curr_price)

		self.timesteps += 1
		curr_state = np.array([curr_state.T])

		#Compute reward 
		if test == False:
			if(self.timesteps is not self.horizon):
				self.current_index += 1
				return curr_state, 0, False
			else:
				reward = (self.long - self.short) * self.unit * curr_price - self.bought + self.sold 
				return curr_state, reward, True 
		else:
			self.current_index += 1
			reward = (self.long - self.short) * self.unit * curr_price - self.bought + self.sold
			return curr_state, reward, True 
			
		return 0













