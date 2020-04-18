import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, temp, observations, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	state_dict = {tags[state]: state for state in range(len(tags))}
	A = np.zeros((len(tags), len(tags)))
	pi = np.zeros(len(tags))
	for i in range(len(train_data)):
		index = state_dict[train_data[i].tags[0]]
		pi[index] += 1
	pi /= sum(pi)
	for i in range(len(train_data)):
		for j in range(train_data[i].length - 1):
			index_x = state_dict[train_data[i].tags[j]]
			index_y = state_dict[train_data[i].tags[j + 1]]
			A[index_x][index_y] += 1

	for i in range(len(tags)):
		if (sum(A[i] != 0)):
			A[i] = A[i] / sum(A[i])
	observations = {}
	m = 0
	for i in range(len(train_data)):
		for word in train_data[i].words:
			if (word not in observations):
				observations[word] = m
				m += 1
	temp = np.zeros((len(tags), m))
	for i in range(len(train_data)):
		for j in range(train_data[i].length):
			index_x = state_dict[train_data[i].tags[j]]
			index_y = observations[train_data[i].words[j]]
			temp[index_x][index_y] += 1
	for i in range(len(tags)):
		if (sum(temp[i] != 0)):
			temp[i] = temp[i] / sum(temp[i])
	model = HMM(pi, A, temp, observations, state_dict)
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	for i,_ in enumerate(test_data):
		m = len(model.observations)
		new_col = np.array([0.000001 for i in range(len(model.state_dict))])
		for word in test_data[i].words:
			if (word not in model.observations):
				model.observations[word] = m
				model.temp = np.insert(model.temp, m, new_col, axis=1)
				m += 1
		tagging.append(model.viterbi(test_data[i].words))
	###################################################
	return tagging
