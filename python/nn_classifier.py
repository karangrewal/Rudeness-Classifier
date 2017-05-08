"""
Predict rudeness using MFCC acceleration values and a 
feed-forward neural network.

Arguments:
0 -- train on MFCC acceleration features
1 -- train on Prosodic features
2 -- train on both
3 -- train on all MFCC features (39 in total)
"""
from __future__ import print_function

from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

import numpy as np
from scipy import stats
import os
import sys

# Global Variables
PATH = '/Users/karangrewal/documents/developer/rudeness-classifier/conversations/'
PYTHON = '/Users/karangrewal/documents/developer/rudeness-classifier/python/'

def smooth(y_test):
	buf_size = 19
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
	if rmin is None:
		rmin = X.min(axis=0)
	if rmax is None:
		rmax = X.max(axis=0)
	N = (X - rmin) / (rmax - rmin)
	N = np.nan_to_num(N)
	return N, rmin, rmax

def mixed_training_data():
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
	X_Y_train = np.int32(X_Y_train)
	return X_Y_train

def mixed_test_data():
	X_Y_test = None
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
			if X_Y_test is not None:
				X_Y_test = np.concatenate((X_Y_test, data), axis=0)
			else:
				X_Y_test = data
	X_Y_test = np.int32(X_Y_test)
	return X_Y_test

def get_training_data(t_arg):
	return mixed_training_data()
	
def get_test_data(t_arg):
	return mixed_test_data()

def apply_bin_labels(X):
	""" Apply binary labels to X. """
	A = np.copy(X)
	label_i = A.shape[1] - 1
	# A[A[:,label_i]==0,label_i] = -1
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

def network(tier):
	if tier == 1:
		inp = Input(shape=(18,))
		out = Dense(32, activation='relu')(inp)
		out = Dense(12, activation='relu')(out)
		out = Dense(2, activation='softmax')(out)
		model = Model(input=inp, output=out)
		model.compile(optimizer=Adam(lr=0.5), loss='categorical_crossentropy', metrics=['accuracy'])
	elif tier == 2:
		inp = Input(shape=(18,))
		out = Dense(12, activation='relu')(inp)
		out = Dense(4, activation='softmax', W_regularizer=l2(0.1), b_regularizer=l2(0.1))(out)
		model = Model(input=inp, output=out)
		model.compile(optimizer=Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def multi_tier_nn(t_arg, normalize=True):
	"""
	Train and make predictions using a two-tier neural network.
	Only supports type 2 features.

	t_arg: which features to train network on
	normalize: scale data values to the interval [0,1]
	"""

	######################## TRAIN TIER-ONE NETWORK ########################

	print('===\n... TRAINING TIER 1 NETWORK ...\n===')
	rmin, rmax = None, None

	# Load training data
	X_Y_train = get_training_data(t_arg)
	
	# Trim examples for each class
	X_Y_train_0 = trim_examples(X_Y_train[X_Y_train[:,-1]==0,:], 10000)
	X_Y_train_1 = trim_examples(X_Y_train[X_Y_train[:,-1]==1,:], 5000)
	X_Y_train_2 = trim_examples(X_Y_train[X_Y_train[:,-1]==2,:], 3200)
	X_Y_train_3 = trim_examples(X_Y_train[X_Y_train[:,-1]==3,:], 1300)
	
	X_Y_train = np.concatenate((X_Y_train_0, X_Y_train_1, X_Y_train_2, X_Y_train_3), axis=0)
	np.random.shuffle(X_Y_train)

	# Apply binary labels
	X_Y_train = apply_bin_labels(X_Y_train)
	
	if normalize:
		X_train, rmin, rmax = scale(X_Y_train[:,:18], rmin, rmax)
	else:
		X_train = X_Y_train[:,:18]
	Y_train = to_categorical(X_Y_train[:,18])
	
	# Train neural network
	model_1 = network(1)
	checkpoint = ModelCheckpoint(os.path.join(PYTHON, 'net1.hdf5'), monitor='loss', save_best_only=True)
	model_1.fit(X_train, Y_train, nb_epoch=50, callbacks=[checkpoint], validation_split=0.1)

	# Load test data
	X_Y_test = get_test_data(t_arg)
	X_Y_test_bin = apply_bin_labels(X_Y_test)

	if normalize:
		X_test, rmin, rmax = scale(X_Y_test_bin[:,:18], rmin, rmax)
	else:
		X_test = X_Y_test_bin[:,:18]
	Y_test = to_categorical(X_Y_test_bin[:,18])

	# Make predictions
	P = model_1.predict(X_test)
	P = np.argmax(P, axis=1)

	# Apply smoothing function
	P_smooth = smooth(P)

	# Discard examples that were not classified as +1
	X_Y_test = np.concatenate((X_Y_test, np.array(P).reshape(-1,1)), axis=1)
	X_Y_test = X_Y_test[X_Y_test[:,-1]>0,:]
	X_Y_test = X_Y_test[:,:-1]

	# Save tier one predictions
	comparison = np.concatenate((np.array(P_smooth).reshape(-1,1), np.array(P).reshape(-1,1), np.argmax(Y_test, axis=1).reshape(-1,1)), axis=1)
	np.savetxt(os.path.join(PYTHON, 'nn_output_tier1.csv'), comparison, delimiter=',')

	######################## TRAIN TIER-TWO NETWORK ########################

	if X_Y_test.shape[0] == 0:
		exit(0)

	print('===\n... TRAINING TIER 2 NETWORK ...\n===')
	rmin, rmax = None, None

	X_Y_train_1 = trim_examples(X_Y_train_1, 3500)
	X_Y_train_2 = trim_examples(X_Y_train_2, 3200)
	X_Y_train_3 = trim_examples(X_Y_train_3, 1300)

	X_Y_train = np.concatenate((X_Y_train_1, X_Y_train_2, X_Y_train_3), axis=0)
	np.random.shuffle(X_Y_train)

	# Training data has already been scaled
	X_train = X_Y_train[:,:18]
	Y_train = to_categorical(X_Y_train[:,18])

	# Train neural network
	model_2 = network(2)
	checkpoint = ModelCheckpoint(os.path.join(PYTHON, 'net2.hdf5'), monitor='loss', save_best_only=True)
	model_2.fit(X_train, Y_train, nb_epoch=50, callbacks=[checkpoint], validation_split=0.1)

	# Test data has already been scaled
	X_test = X_Y_test[:,:18]
	Y_test = to_categorical(X_Y_test[:,18])

	# Make predictions
	P = model_2.predict(X_test)
	P = np.argmax(P, axis=1)

	# Apply smoothing function
	P_smooth = smooth(P)

	# Save tier one predictions
	comparison = np.concatenate((np.array(P_smooth).reshape(-1,1), np.array(P).reshape(-1,1), np.argmax(Y_test, axis=1).reshape(-1,1)), axis=1)
	np.savetxt(os.path.join(PYTHON, 'nn_output_tier2.csv'), comparison, delimiter=',')

if __name__ == '__main__':

	# Test Examples
	data = dict({1:[14, 15, 17, 18, 19, 21, 26, 27, 29, 30, 31, 32, 34, 37, 38, 41, 54, 56, 58, 60, 63, 64, 66, 69, 70, 71, 73, 75, 76, 77, 78],
		2:[12, 13, 22, 23, 35, 46, 51, 53, 55],
		3:[16, 20, 24, 28, 39, 40, 42, 44, 47]})

	test_cases = list()
	test_cases.append('rude_%d.csv' % np.random.choice(data[1]))
	test_cases.append('rude_%d.csv' % np.random.choice(data[2]))
	test_cases.append('rude_%d.csv' % np.random.choice(data[3]))

	# if len(sys.argv) != 2:
	# 	exit(0)
	
	# t_arg = int(sys.argv[1])
	multi_tier_nn(2)
