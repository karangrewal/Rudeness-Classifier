'''
NOTE: original file ~/downloads/LIBSVM-3.21/python/classify.py

usage: `python classify.py <reuse>`

mfcc: train SVM classifier using MFCC values.
mfcc_de: train SVM classifier using MFCC delta (velocity) values.
mfcc_de_de: train SVM classifier using MFCC delta delta (acceleration) values.

Arguments:


'''

import numpy as np
from scipy import stats
from svmutil import *
import os
import sys

# Smoothing buffer size
B = 21
LIBSVM = '/Users/karangrewal/downloads/libsvm-3.21/python'
PATH = '/Users/karangrewal/documents/developer/rudeness-classifier/conversations/'

# Test Examples
TEST_CASES = ['rude_37.csv', 'rude_49.csv', 'rude_51.csv', 'rude_72.csv']

def smooth(y_test, buf_size):
	buf_size = 1. * buf_size / 2
	y_test = np.array(y_test)
	smooth = np.zeros(y_test.shape[0])

	for i in range(y_test.shape[0]):
		smooth[i] = stats.mode(y_test[max([i - 3,0]):min([i + 3,y_test.shape[0] - 1])])[0][0]

	return smooth

def get_training_data():
	X_train, Y_train = None, None

	example_files = os.listdir(os.path.join(PATH, 'frames'))
	for f in example_files:
		if f.endswith('.csv') and f not in TEST_CASES:

			# Extract all examples
			print('Training on %s' % f)
			
			file_data = np.genfromtxt(os.path.join(PATH, 'frames', f), delimiter=',', skip_header=0)
			label_index = file_data.shape[1] - 1

			if X_train is not None:
				X_train = np.concatenate((X_train, file_data[:,:label_index]), axis=0)
				Y_train = np.concatenate((Y_train, file_data[:,label_index]), axis=0)
			else:
				X_train = file_data[:,:label_index]
				Y_train = file_data[:,label_index]

	Y_train = np.float32(Y_train)
	for i in range(Y_train.shape[0]):
		if Y_train[i] == 0:
			Y_train[i] = -1.
		else:
			Y_train[i] = 1.
	return X_train, Y_train

def get_test_data():
	X_test, Y_test = None, None

	test_files = os.listdir(os.path.join(PATH, 'frames'))
	for f in test_files:
		if f in TEST_CASES and f.endswith('.csv'):

			file_data = np.genfromtxt(os.path.join(PATH, 'frames', f), delimiter=',', skip_header=0)
			label_index = file_data.shape[1] - 1

			if X_test is not None:
				X_test = np.concatenate((X_test, file_data[:,:label_index]), axis=0)
				Y_test = np.concatenate((Y_test, file_data[:,label_index]), axis=0)
			else:
				X_test = file_data[:,:label_index]
				Y_test = file_data[:,label_index]

	Y_test = np.float32(Y_test)
	for i in range(Y_test.shape[0]):
		if Y_test[i] == 0:
			Y_test[i] = -1.
		else:
			Y_test[i] = 1.
	return X_test, Y_test

def mfcc_de_de(reuse):
	

	if reuse:
		# Load SVM model from memory
		model = svm_load_model(os.path.join(LIBSVM, 'svm_de_de.model'))

	else:
		# Train SVM classifier and make predictions
		# Get Training Data
		X_train, Y_train = get_training_data()
		X_train = X_train[:,28:40].reshape(-1, 12)

		# Convert to python standard data types
		X_train = np.ndarray.tolist(X_train)
		Y_train = np.ndarray.tolist(Y_train)

		print(Y_train)

		model = svm_train(Y_train, X_train)

	# Get Test Data
	X_test, Y_test = get_test_data()
	X_test = X_test[:,28:40].reshape(-1, 12)
	X_test = np.ndarray.tolist(X_test)
	Y_test = np.ndarray.tolist(Y_test)

	# Make predictions
	p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)
	np.savetxt(os.path.join(LIBSVM, 'stats.csv'), Y_test, delimiter=',')

	# Apply smoothing function
	p_label_smooth = smooth(p_label, B)

	# Save model
	svm_save_model(os.path.join(LIBSVM, 'svm_de_de.model'), model)

	# Save predictions
	comparison = np.concatenate((np.array(p_label_smooth).reshape(-1,1), np.array(p_label).reshape(-1,1), np.array(Y_test).reshape(-1,1)), axis=1)
	np.savetxt(os.path.join(LIBSVM, 'labels.csv'), comparison, delimiter=',')

def mfcc_de():
	# Get Training Data
	X_train, Y_train = get_training_data()
	X_train = X_train[:,15:27].reshape(-1, 12)

	# Get Test Data
	X_test, Y_test = get_test_data()
	X_test = X_test[:,15:27].reshape(-1, 12)

	# Convert to python standard data types
	X_train = np.ndarray.tolist(X_train)
	X_test = np.ndarray.tolist(X_test)
	Y_train = np.ndarray.tolist(Y_train)
	Y_test = np.ndarray.tolist(Y_test)

	# Train SVM classifier and make predictions
	model = svm_train(Y_train, X_train)
	p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)

	# Save predictions
	p_label = np.array(p_label)
	comparison = np.concatenate((np.array(p_label).reshape(-1,1), np.array(Y_test).reshape(-1,1)), axis=1)
	np.savetxt(os.path.join(PATH, 'frames', 'labels.csv'), comparison, delimiter=',')

def mfcc():
	# Get Training Data
	X_train, Y_train = get_training_data()
	X_train = X_train[:,2:14].reshape(-1, 12)

	# Get Test Data
	X_test, Y_test = get_test_data()
	X_test = X_test[:,2:14].reshape(-1, 12)

	# Convert to python standard data types
	X_train = np.ndarray.tolist(X_train)
	X_test = np.ndarray.tolist(X_test)
	Y_train = np.ndarray.tolist(Y_train)
	Y_test = np.ndarray.tolist(Y_test)

	# Train SVM classifier and make predictions
	model = svm_train(Y_train, X_train)
	p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)

	# Save predictions
	p_label = np.array(p_label)
	np.savetxt(os.path.join(PATH, 'frames', 'labels.csv'), p_label, delimiter=',')

if __name__ == '__main__':
	if len(sys.argv) > 1:
		mfcc_de_de(int(sys.argv[1]))
	else:
		mfcc_de_de(0)
