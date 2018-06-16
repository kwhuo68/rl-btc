import numpy as np
import math
import random

class ExperienceReplay:
	def __init__(self, memory_size = 50, discount = 0.9):
		self.memory_size = memory_size
		#Elements -> [state_t, action_t, reward_t, state_t+1], termination
		self.memory = list() 
		self.discount = discount

	def store(self, states, termination):
		#Each state is: s_t, a_t, r_t, s_{t+1}
		self.memory.append([states, termination])
		if(len(self.memory) > self.memory_size):
			del self.memory[0]

	def size(self):
		return len(self.memory)

	def sample_batch(self, model, batch_size):
		num_features = 6
		num_actions = 3
		sample_num = min(len(self.memory), batch_size)

		#input: (s, a) pairs, target: reward values 
		input_batch = [] 
		target_batch = [] 
		random_sample = random.sample(self.memory, sample_num)
		iteration = 0

		for sample in random_sample:
			state_t, action_t, reward_t, state_t_plus_1 = sample[0]
			Q_sa_orig = model.predict(state_t, batch_size = 1)
			termination = sample[1]

			#Predicted Q-value
			Q_sa_pred = np.max(model.predict(state_t_plus_1, batch_size = 1))
			target_vals = np.zeros((1, 3))
			target_vals[:] = Q_sa_orig[0]

			if termination: 
				target_vals[0][action_t] = reward_t
			else:
				target_vals[0][action_t] = reward_t + self.discount * Q_sa_pred

			input_batch.append(state_t.reshape(num_features, ))
			target_batch.append(target_vals.reshape(num_features, ))
			iteration += 1

		input_batch = np.array(input_batch)
		target_batch = np.array(target_batch)

		return input_batch, target_batch
