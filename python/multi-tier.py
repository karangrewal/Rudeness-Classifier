"""
Note: original file ~/downloads/LIBSVM-3.21/python/multi-tier.py

Train a two-tier SVM classifier using MFCC and Prosodic featueres.

Arguments:
*0 -- train on MFCC acceleration features
*1 -- train on Prosodic features
 2 -- train on both
*3 -- train on all MFCC features (39 in total)

* : unsupported
"""

import numpy as np
from svmutil import *

import os
import sys

from classify import LIBSVM, PATH
from classify import apply_bin_labels, get_training_data, get_test_data, scale, smooth, trim_examples

def train_multi_tier(t_arg, normalize=True):
	"""
	Train and make predictions using a two-tier model.
	Only supports type 2 features.

	t_arg: which features to train SVM on
	normalize: scale data values to the interval [0,1]
	"""
	
	######################## TRAIN TIER-ONE SVM ########################

	print('===\n... TRAINING TIER 1 CLASSIFIER ...\n===')
	rmin, rmax = None, None

	# Load training data
	X_Y_train = get_training_data(t_arg, test_cases)

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
	model_1 = svm_train(Y_train, X_train)#, '-g 0.5')
	svm_save_model(os.path.join(LIBSVM, 'svm_tier1.model'), model_1)

	# Load test data
	X_Y_test = get_test_data(t_arg, test_cases)
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

	######################## TRAIN TIER-TWO SVM ########################
	
	print('===\n... TRAINING TIER 2 CLASSIFIER ...\n===')
	rmin, rmax = None, None

	X_Y_train_1 = trim_examples(X_Y_train_1, 3500)
	X_Y_train_2 = trim_examples(X_Y_train_2, 3200)
	X_Y_train_3 = trim_examples(X_Y_train_3, 1300)
	
	X_Y_train = np.concatenate((X_Y_train_1, X_Y_train_2, X_Y_train_3), axis=0)
	np.random.shuffle(X_Y_train)

	# Training data has already been scaled
	X_train = np.ndarray.tolist(X_Y_train[:,:18])
	Y_train = np.ndarray.tolist(X_Y_train[:,18])

	# Train tier-two SVM
	model_2 = svm_train(Y_train, X_train)
	svm_save_model(os.path.join(LIBSVM, 'svm_tier2.model'), model_2)
	
	# Test data has already been scaled
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

	# if len(sys.argv) != 2:
	# 	exit(0)
	
	# t_arg = int(sys.argv[1])
	train_multi_tier(2)
