from processor import Processor
from environment import Environment
from experience_replay import ExperienceReplay
from agent import Agent

def main():
	hist_length = 50
	processor = Processor(history_length = hist_length)
	price_history = processor.fetchData()
	train_price_history = price_history['train']
	test_price_history = price_history['test']
	env = Environment(horizon = 20, train_price_history = train_price_history, test_price_history = test_price_history, history_length = hist_length)
	exp_replay = ExperienceReplay()
	agent = Agent(feature_size = 6, window = hist_length, action_size = 3, experience_replay = exp_replay, environment = env)
	agent.train()
	print("Agent done training, now testing: ")
	agent.test(test_price_history)

if __name__ == '__main__':
    main()