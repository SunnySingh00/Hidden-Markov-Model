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
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	state_dict = {tags[state]: state for state in range(len(tags))}
	print(state_dict)
	print("-----------------")

	# for x in train_data:
	# print(x.id,x.length,x.words,x.tags)

	no_states = len(tags)
	A = np.zeros((no_states, no_states))
	pi = np.zeros(no_states)

	n = len(train_data)
	for i in range(n):
		index = state_dict[train_data[i].tags[0]]
		pi[index] += 1

	total_pi = sum(pi)
	pi = pi / total_pi

	print(pi)
	print("-------------------")
	for i in range(n):
		for j in range(train_data[i].length - 1):
			index_x = state_dict[train_data[i].tags[j]]
			index_y = state_dict[train_data[i].tags[j + 1]]
			A[index_x][index_y] += 1

	# print(A)
	for i in range(no_states):
		if (sum(A[i] != 0)):
			A[i] = A[i] / sum(A[i])
	print(A)
	print("-------------------")

	obs_dict = {}
	m = 0
	for i in range(n):
		for word in train_data[i].words:
			if (word not in obs_dict):
				obs_dict[word] = m
				m += 1
	print(m, obs_dict)
	print("-------------------")

	B = np.zeros((no_states, m))
	for i in range(n):
		# print(train_data[i].words)
		for j in range(train_data[i].length):
			index_x = state_dict[train_data[i].tags[j]]
			index_y = obs_dict[train_data[i].words[j]]
			B[index_x][index_y] += 1
	# print(B)
	for i in range(no_states):
		if (sum(B[i] != 0)):
			B[i] = B[i] / sum(B[i])
	print(B)

	model = HMM(pi, A, B, obs_dict, state_dict)
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
	for s in range(len(test_data)):
		m = len(model.obs_dict)
		# print("before",len(model.obs_dict),model.obs_dict)
		new_col = np.array([0.000001 for i in range(len(model.state_dict))])
		for word in test_data[s].words:
			if (word not in model.obs_dict):
				model.obs_dict[word] = m
				model.B = np.insert(model.B, m, new_col, axis=1)
				m += 1
		# print("after",len(model.obs_dict),model.obs_dict)
		viterbi_path = model.viterbi(test_data[s].words)
		# print(test_data[0].tags)
		# print(viterbi_path)
		tagging.append(viterbi_path)
	###################################################
	return tagging
