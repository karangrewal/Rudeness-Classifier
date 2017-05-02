"""
Note: original file ~/downloads/LIBSVM-3.21/python/classify.py

New Classify.py: train SVM based on:
- MFCC acceleration features
- Prosodic features
- Both

Arguments:
0 -- train on MFCC acceleration features
1 -- train on Prosodic features
2 -- train on both
3 -- train on all MFCC features (39 in total)
"""

import numpy as np
from scipy import stats
from svmutil import *

import os
import sys

# Global Variables
LIBSVM = '<path-to-libsvm-folder>'
PATH = '<path-to-local-folder>'

def smooth(y_test):
	buf_size = 39
	buf_size = 1. * buf_size / 2
	y_test = np.array(y_test)
	smooth = np.zeros(y_test.shape[0])

	for i in range(y_test.shape[0]):
		smooth[i] = stats.mode(y_test[max([i - 3,0]):min([i + 3,y_test.shape[0] - 1])])[0][0]

	return smooth

def scale(X, rmin=None, rmax=None, type_scale=1):
	"""
	Two types of normalizations:
	1 -- Each feature is scaled to a value between 0 and 1
	2 -- Each example vector has unit norm
	"""
	# TYPE 2
	# 	norms = np.linalg.norm(X, axis=1)
	# 	norms = norms.reshape(-1, norms.shape[0])
	# 	N = X / norms.T

	if rmin is None:
		rmin = X.min(axis=0)
	if rmax is None:
		rmax = X.max(axis=0)
	N = (X - rmin) / (rmax - rmin)
	N = np.nan_to_num(N)
	return N, rmin, rmax

def mixed_training_data(bin_classification=False):
	X_Y_train = None
	path1 = os.path.join(PATH, 'mfcc')
	path2 = os.path.join(PATH, 'prosody')

	for i in range(12, 79):
		if ('rude_%d.csv' % i) not in test_cases:
			#print('training on rude_%d.csv' % i)
			mfcc_data = np.genfromtxt(os.path.join(path1, 'rude_%d.csv' % i), delimiter=',', skip_header=0)
			mfcc_data = mfcc_data[:,28:]
			prosody_data = np.genfromtxt(os.path.join(path2, 'rude_%d.csv' % i), delimiter=',', skip_header=0)
			prosody_data = prosody_data[:,1:]
			if mfcc_data.shape[0] < prosody_data.shape[0]:
				prosody_data = prosody_data[1:,:]
			assert mfcc_data.shape[0] == prosody_data.shape[0]

			data = np.concatenate((mfcc_data[:,:-1], prosody_data),axis=1)
			if X_Y_train is not None:
				X_Y_train = np.concatenate((X_Y_train, data), axis=0)
			else:
				X_Y_train = data
	X_Y_train = np.float32(X_Y_train)

	return X_Y_train

def mixed_test_data():
	X_Y_train = None
	path1 = os.path.join(PATH, 'mfcc')
	path2 = os.path.join(PATH, 'prosody')

	for i in range(12, 79):
		if ('rude_%d.csv' % i) in test_cases:
			mfcc_data = np.genfromtxt(os.path.join(path1, 'rude_%d.csv' % i), delimiter=',', skip_header=0)
			mfcc_data = mfcc_data[:,28:]
			prosody_data = np.genfromtxt(os.path.join(path2, 'rude_%d.csv' % i), delimiter=',', skip_header=0)
			prosody_data = prosody_data[:,1:]
			if mfcc_data.shape[0] < prosody_data.shape[0]:
				prosody_data = prosody_data[1:,:]
			assert mfcc_data.shape[0] == prosody_data.shape[0]

			data = np.concatenate((mfcc_data[:,:-1], prosody_data),axis=1)
			if X_Y_train is not None:
				X_Y_train = np.concatenate((X_Y_train, data), axis=0)
			else:
				X_Y_train = data
	X_Y_train = np.float32(X_Y_train)

	return X_Y_train

def get_training_data(t_arg):
	X_Y_train, t_path = None, None

	if t_arg == 0 or t_arg == 3:
		example_files = os.listdir(os.path.join(PATH, 'mfcc'))
		t_path = os.path.join(PATH, 'mfcc')
	elif t_arg == 1:
		example_files = os.listdir(os.path.join(PATH, 'prosody'))
		t_path = os.path.join(PATH, 'prosody')
	elif t_arg == 2:
		return mixed_training_data()
	else:
		exit(0)

	for f in example_files:
		if f.endswith('.csv') and f not in test_cases:

			#print('training on %s' % f)
			file_data = np.genfromtxt(os.path.join(t_path, f), delimiter=',', skip_header=0)

			if X_Y_train is not None:
				X_Y_train = np.concatenate((X_Y_train, file_data), axis=0)
			else:
				X_Y_train = file_data
	X_Y_train = np.float32(X_Y_train)

	if t_arg == 0:
		X_Y_train = X_Y_train[:,28:].reshape(-1,13)
	elif t_arg == 1:
		X_Y_train = X_Y_train[:,1:].reshape(-1,7)
	elif t_arg == 3:
		X_Y_train = X_Y_train[:,1:].reshape(-1,40)

	return X_Y_train

def get_test_data(t_arg):
	X_Y_test, t_path = None, None

	if t_arg == 0 or t_arg == 3:
		example_files = os.listdir(os.path.join(PATH, 'mfcc'))
		t_path = os.path.join(PATH, 'mfcc')
	elif t_arg == 1:
		example_files = os.listdir(os.path.join(PATH, 'prosody'))
		t_path = os.path.join(PATH, 'prosody')
	elif t_arg == 2:
		return mixed_test_data()

	for f in example_files:
		if f.endswith('.csv') and f in test_cases:

			file_data = np.genfromtxt(os.path.join(t_path, f), delimiter=',', skip_header=0)

			if X_Y_test is not None:
				X_Y_test = np.concatenate((X_Y_test, file_data), axis=0)
			else:
				X_Y_test = file_data
	X_Y_test = np.float32(X_Y_test)

	if t_arg == 0:
		X_Y_test = X_Y_test[:,28:].reshape(-1,13)
	elif t_arg == 1:
		X_Y_test = X_Y_test[:,1:].reshape(-1,7)
	elif t_arg == 3:
		X_Y_test = X_Y_test[:,1:].reshape(-1,40)

	return X_Y_test

def apply_bin_labels(X):
	""" Apply binary labels to X. """
	A = np.copy(X)
	label_i = A.shape[1] - 1
	A[A[:,label_i]==0,label_i] = -1
	A[A[:,label_i]>0,label_i] = 1
	return A

def trim_examples(A, target):
	""" Randomly remove rows from A (2D numpy array) until it only contains sample rows. """
	if A.shape[0] > target:
		while A.shape[0] > target:
			num_to_del = A.shape[0] - target
			t = np.random.randint(0, A.shape[0], num_to_del)
			A = np.delete(A, t, axis=0)
	return A

def train_direct(t_arg, reuse=False, normalize=True, bin_classification=False):
	"""
	Train and make predictions using a single-tier model.

	t_arg: which features to train SVM on
	reuse: load most recently trained model iff True
	normalize: scale data values to the interval [0,1]
	bin_classification: use binary -1/+1 labels
	"""
	rmin, rmax = None, None

	if reuse:
		model = svm_load_model(os.path.join(LIBSVM, 'svm.model'))
	else:
		# Load training data
		X_Y_train = get_training_data(t_arg)
		
		# Trim examples for each class
		X_Y_train_0 = trim_examples(X_Y_train[X_Y_train[:,-1]==0,:], 15000)
		X_Y_train_1 = trim_examples(X_Y_train[X_Y_train[:,-1]==1,:], 4000)
		X_Y_train_2 = trim_examples(X_Y_train[X_Y_train[:,-1]==2,:], 2500)
		X_Y_train_3 = trim_examples(X_Y_train[X_Y_train[:,-1]==3,:], 1500)

		X_Y_train = np.concatenate((X_Y_train_0, X_Y_train_1, X_Y_train_2, X_Y_train_3), axis=0)
		np.random.shuffle(X_Y_train)

		# Apply binary labels
		if bin_classification:
			X_Y_train = apply_bin_labels(X_Y_train)

		# Convert to python standard data types
		if t_arg == 0:
			X_train = np.ndarray.tolist(X_Y_train[:,:12])
			Y_train = np.ndarray.tolist(X_Y_train[:,12])
		elif t_arg == 1:
			if normalize:
				X_train, rmin, rmax = scale(X_Y_train[:,:6])
				X_train = np.ndarray.tolist(X_train)
			else:
				X_train = np.ndarray.tolist(X_Y_train[:,:6])
			Y_train = np.ndarray.tolist(X_Y_train[:,6])
		elif t_arg == 2:
			if normalize:
				X_train, rmin, rmax = scale(X_Y_train[:,:18], rmin, rmax)
				X_train = np.ndarray.tolist(X_train)
			else:
				X_train = np.ndarray.tolist(X_Y_train[:,:18])
			Y_train = np.ndarray.tolist(X_Y_train[:,18])
		elif t_arg == 3:
			X_train = np.ndarray.tolist(X_Y_train[:,:39])
			Y_train = np.ndarray.tolist(X_Y_train[:,39])

		# Train SVM
		model = svm_train(Y_train, X_train)#, '-t 2 -g 0.5 -c 1')
		svm_save_model(os.path.join(LIBSVM, 'svm.model'), model)

	# Load test data
	X_Y_test = get_test_data(t_arg)
	if bin_classification:
		X_Y_test = apply_bin_labels(X_Y_test)

	if t_arg == 0:
		X_test = np.ndarray.tolist(X_Y_test[:,:12])
		Y_test = np.ndarray.tolist(X_Y_test[:,12])
	elif t_arg == 1:
		if normalize:
			X_test, rmin, rmax = scale(X_Y_test[:,:6], rmin, rmax)
			X_test = np.ndarray.tolist(X_test)
		else:
			X_test = np.ndarray.tolist(X_Y_test[:,:6])
		Y_test = np.ndarray.tolist(X_Y_test[:,6])
	elif t_arg == 2:
		if normalize:
			X_test, rmin, rmax = scale(X_Y_test[:,:18], rmin, rmax)
			X_test = np.ndarray.tolist(X_test)
		else:
			X_test = np.ndarray.tolist(X_Y_test[:,:18])
		Y_test = np.ndarray.tolist(X_Y_test[:,18])
	elif t_arg == 3:
		X_test = np.ndarray.tolist(X_Y_test[:,:39])
		Y_test = np.ndarray.tolist(X_Y_test[:,39])

	# Make predictions using trained model
	p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)
	
	# Apply smoothing function
	p_label_smooth = smooth(p_label)

	# Save predictions
	comparison = np.concatenate((np.array(p_label_smooth).reshape(-1,1), np.array(p_label).reshape(-1,1), np.array(Y_test).reshape(-1,1)), axis=1)
	np.savetxt(os.path.join(LIBSVM, 'output_%d.csv' % t_arg), comparison, delimiter=',')

def train_multi_tier(t_arg, normalize=True):
	"""
	Train and make predictions using a two-tier model.
	Only supports type 2 features.

	t_arg: which features to train SVM on
	normalize: scale data values to the interval [0,1]
	"""
	if t_arg != 2:
		exit(0)

	####################################################
	################ TRAIN TIER-ONE SVM ################
	####################################################
	print('===\n... TRAINING TIER 1 CLASSIFIER ...\n===')
	rmin, rmax = None, None

	# Load training data
	X_Y_train = get_training_data(t_arg)

	# Trim examples for each class
	X_Y_train_0 = trim_examples(X_Y_train[X_Y_train[:,-1]==0,:], 15000)
	X_Y_train_1 = trim_examples(X_Y_train[X_Y_train[:,-1]==1,:], 5000)
	X_Y_train_2 = trim_examples(X_Y_train[X_Y_train[:,-1]==2,:], 3200)
	X_Y_train_3 = trim_examples(X_Y_train[X_Y_train[:,-1]==3,:], 1300)

	X_Y_train = np.concatenate((X_Y_train_0, X_Y_train_1, X_Y_train_2, X_Y_train_3), axis=0)
	np.random.shuffle(X_Y_train)

	# Apply binary labels
	X_Y_train = apply_bin_labels(X_Y_train)

	# Convert to python standard data types
	if normalize:
		X_train, rmin, rmax = scale(X_Y_train[:,:-1], rmin, rmax)
		X_train = np.ndarray.tolist(X_train)
	else:
		X_train = np.ndarray.tolist(X_Y_train[:,:-1])
	Y_train = np.ndarray.tolist(X_Y_train[:,-1])
	
	# Train tier-one SVM
	model_1 = svm_train(Y_train, X_train, '-g 0.5')
	svm_save_model(os.path.join(LIBSVM, 'svm_tier1.model'), model_1)

	# Load test data
	X_Y_test = get_test_data(t_arg)
	
	# Assign indices for future reference of individual training points
	#X_Y_test = np.concatenate((X_Y_test[:,:-1], np.arange(X_Y_test.shape[0]).reshape(-1,1), X_Y_test[:,-1].reshape(-1,1)), axis=1)
	
	# Binary labels
	X_Y_test_bin = apply_bin_labels(X_Y_test)
	
	if normalize:
		X_test, rmin, rmax = scale(X_Y_test_bin[:,:-1], rmin, rmax)
		X_test = np.ndarray.tolist(X_test)
	else:
		X_test = np.ndarray.tolist(X_Y_test_bin[:,:-1])
	Y_test = np.ndarray.tolist(X_Y_test_bin[:,-1])

	# Make predictions using trained model
	p_label, p_acc, p_val = svm_predict(Y_test, X_test, model_1)

	# Apply smoothing function
	p_label_smooth = smooth(p_label)
	
	# Only keep examples that were classified as +1
	X_Y_test = np.concatenate((X_Y_test, np.array(p_label).reshape(-1,1)), axis=1)
	X_Y_test = X_Y_test[X_Y_test[:,-1]>0,:]
	X_Y_test = X_Y_test[:,:-1]

	# Save predictions
	comparison = np.concatenate((np.array(p_label_smooth).reshape(-1,1), np.array(p_label).reshape(-1,1), np.array(Y_test).reshape(-1,1)), axis=1)
	np.savetxt(os.path.join(LIBSVM, 'output_tier1.csv'), comparison, delimiter=',')

	####################################################
	################ TRAIN TIER-TWO SVM ################
	####################################################
	print('===\n... TRAINING TIER 2 CLASSIFIER ...\n===')
	rmin, rmax = None, None
	X_Y_train_1 = trim_examples(X_Y_train_1, 3500)
	X_Y_train_2 = trim_examples(X_Y_train_2, 3200)
	X_Y_train_3 = trim_examples(X_Y_train_3, 1300)
	X_Y_train = np.concatenate((X_Y_train_1, X_Y_train_2, X_Y_train_3), axis=0)
	np.random.shuffle(X_Y_train)

	# s = X_Y_train_1.shape[0] + X_Y_train_2.shape[0] + X_Y_train_3.shape[0]
	# print(1. * X_Y_train_1.shape[0] / s)
	# print(1. * X_Y_train_2.shape[0] / s)
	# print(1. * X_Y_train_3.shape[0] / s)	

	# Convert to python standard data types
	# if normalize:
	# 	X_train, rmin, rmax = scale(X_Y_train[:,:18], rmin, rmax)
	# 	X_train = np.ndarray.tolist(X_train)
	# else:
	X_train = np.ndarray.tolist(X_Y_train[:,:18])
	Y_train = np.ndarray.tolist(X_Y_train[:,18])

	np.savetxt(os.path.join(LIBSVM, 'y_train_2.csv'), X_Y_train[:,18], delimiter=',')

	# Train tier-two SVM
	model_2 = svm_train(Y_train, X_train)
	svm_save_model(os.path.join(LIBSVM, 'svm_tier2.model'), model_2)
	
	# Already normalized data
	# if normalize:
	# 	X_test, rmin, rmax = scale(X_Y_test[:,:-1], rmin, rmax)
	# 	X_test = np.ndarray.tolist(X_test)
	# else:
	X_test = np.ndarray.tolist(X_Y_test[:,:-1])
	Y_test = np.ndarray.tolist(X_Y_test[:,-1])
	
	# Make predictions using tier-two SVM
	p_label, p_acc, p_val = svm_predict(Y_test, X_test, model_2)
	
	# Apply smoothing function
	p_label_smooth = smooth(p_label)

	# Save predictions
	comparison = np.concatenate((np.array(p_label_smooth).reshape(-1,1), np.array(p_label).reshape(-1,1), np.array(Y_test).reshape(-1,1)), axis=1)
	np.savetxt(os.path.join(LIBSVM, 'output_tier2.csv'), comparison, delimiter=',')
	
if __name__ == '__main__':

	# Test Examples
	data = dict({1:[14, 15, 17, 18, 19, 21, 26, 27, 29, 30, 31, 32, 34, 37, 38, 41, 54, 56, 58, 60, 63, 64, 66, 69, 70, 71, 73, 75, 76, 77, 78],
		2:[12, 13, 22, 23, 35, 46, 51, 53, 55],
		3:[16, 20, 24, 28, 39, 40, 42, 44, 47]})

	test_cases = list()
	test_cases.append('rude_%d.csv' % np.random.choice(data[1]))
	test_cases.append('rude_%d.csv' % np.random.choice(data[2]))
	test_cases.append('rude_%d.csv' % np.random.choice(data[3]))

	if len(sys.argv) != 2:
		exit(0)
	
	t_arg = int(sys.argv[1])
	#train_direct(t_arg, bin_classification=True)
	train_multi_tier(t_arg)