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
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

import numpy as np
from scipy import stats
import os
import sys

# Global Variables
PATH = '<path-to-local-folder>'
PYTHON = '<path-to-executable-folder>'

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
			print('training on rude_%d.csv' % i)
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

	# Binary Classification
	if bin_classification:
		label_i = X_Y_train.shape[1] - 1
		for i in range(X_Y_train.shape[0]):
			if Y_train[i,label_i] == 0:
				Y_train[i,label_i] = -1.
			else:
				Y_train[i,label_i] = 1.
	return X_Y_train

def mixed_test_data(bin_classification=False):
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

	# Binary Classification
	if bin_classification:
		label_i = X_Y_train.shape[1] - 1
		for i in range(X_Y_train.shape[0]):
			if Y_train[i,label_i] == 0:
				Y_train[i,label_i] = -1.
			else:
				Y_train[i,label_i] = 1.
	return X_Y_train

def get_training_data(t_arg, bin_classification=False):
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

			print('training on %s' % f)
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

	# Binary Classification
	if bin_classification:
		label_i = X_Y_train.shape[1] - 1
		for i in range(X_Y_train.shape[0]):
			if Y_train[i,label_i] == 0:
				Y_train[i,label_i] = -1.
			else:
				Y_train[i,label_i] = 1.
	return X_Y_train

def get_test_data(t_arg, bin_classification=False):
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

	# Binary Classification
	if bin_classification:
		label_i = X_Y_test.shape[1] - 1
		for i in range(X_Y_test.shape[0]):
			if Y_test[i,label_i] == 0:
				Y_test[i,label_i] = -1.
			else:
				Y_test[i,label_i] = 1.
	return X_Y_test

def from_categorical(Y):
	""" Convert 1-of-k encoding to categorical values. """
	L = np.where(Y[:,1]==1., 1., 0.)
	L += np.where(Y[:,2]==1., 2., 0.)
	L += np.where(Y[:,3]==1., 3., 0.)
	return L

def trim_examples(A, target):
	""" Randomly remove rows from A (2D numpy array) until it only contains sample rows. """
	if A.shape[0] > target:
		while A.shape[0] > target:
			num_to_del = A.shape[0] - target
			t = np.random.randint(0, A.shape[0], num_to_del)
			A = np.delete(A, t, axis=0)
	return A

def network(t_arg):
	if t_arg == 0:
		inp = Input(shape=(12,))
	elif t_arg == 1:
		inp = Input(shape=(6,))
	elif t_arg == 2:
		inp = Input(shape=(18,))
	elif t_arg == 3:
		inp = Input(shape=(39,))

	out = Dense(12, activation='relu', W_regularizer=l2(0.05), b_regularizer=l2(0.05))(inp)
	#out = Dense(30, activation='relu')(out)
	#out = Dense(12, activation='relu')(out)
	out = Dense(4, activation='softmax', W_regularizer=l2(0.05), b_regularizer=l2(0.05))(out)

	model = Model(input=inp, output=out)
	model.compile(optimizer=SGD(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

	return model

def main(t_arg, reuse=False, normalize=True):
	"""
	t_arg: which features to train network on.
	reuse: load most recently trained model iff True
	normalize: scale data values to the interval [0,1]
	"""
	rmin, rmax = None, None

	if reuse:
		pass
	else:
		# Load training data
		X_Y_train = get_training_data(t_arg)
		
		# Trim examples for each class
		X_Y_train_0 = trim_examples(X_Y_train[X_Y_train[:,-1]==0,:], 4000)
		X_Y_train_1 = trim_examples(X_Y_train[X_Y_train[:,-1]==1,:], 3500)
		X_Y_train_2 = trim_examples(X_Y_train[X_Y_train[:,-1]==2,:], 2500)
		X_Y_train_3 = trim_examples(X_Y_train[X_Y_train[:,-1]==3,:], 1500)
		X_Y_train = np.concatenate((X_Y_train_0, X_Y_train_1, X_Y_train_2, X_Y_train_3), axis=0)
		
		if t_arg == 0:
			X_train = X_Y_train[:,:12]
			Y_train = to_categorical(X_Y_train[:,12])
		elif t_arg == 1:
			if normalize:
				X_train, rmin, rmax = scale(X_Y_train[:,:6], rmin, rmax)
			else:
				X_train = X_Y_train[:,:6]
			Y_train = to_categorical(X_Y_train[:,6])
		elif t_arg == 2:
			if normalize:
				X_train, rmin, rmax = scale(X_Y_train[:,:18], rmin, rmax)
			else:
				X_train = X_Y_train[:,:18]
			Y_train = to_categorical(X_Y_train[:,18])
		elif t_arg == 3:
			X_train = X_Y_train[:,:39]
			Y_train = to_categorical(X_Y_train[:,39])

		# Train neural network
		model = network(t_arg)
		checkpoint = ModelCheckpoint(os.path.join(PYTHON, 'net.hdf5'), monitor='loss', save_best_only=True)
		model.fit(X_train, Y_train, nb_epoch=50, callbacks=[checkpoint], validation_split=0.1)

	# Load test data
	X_Y_test = get_test_data(t_arg)
	if t_arg == 0:
		X_test = X_Y_test[:,:12]
		Y_test = to_categorical(X_Y_test[:,12])
	elif t_arg == 1:
		if normalize:
			X_test, rmin, rmax = scale(X_Y_test[:,:6], rmin, rmax)
		else:
			X_test = X_Y_test[:,:6]
		Y_test = to_categorical(X_Y_test[:,6])
	elif t_arg == 2:
		if normalize:
			X_test, rmin, rmax = scale(X_Y_test[:,:18], rmin, rmax)
		else:
			X_test = X_Y_test[:,:18]
		Y_test = to_categorical(X_Y_test[:,18])
	elif t_arg == 3:
		X_test = X_Y_test[:,:39]
		Y_test = to_categorical(X_Y_test[:,39])

	# Make predictions
	P = model.predict(X_test)
	P = from_categorical(P)

	# Apply smoothing function
	P_smooth = smooth(P)

	# Save predictions
	comparison = np.concatenate((np.array(P_smooth).reshape(-1,1), np.array(P).reshape(-1,1), from_categorical(Y_test).reshape(-1,1)), axis=1)
	np.savetxt(os.path.join(PYTHON, 'nn_output_%d.csv' % t_arg), comparison, delimiter=',')

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
	main(t_arg, reuse=False)
