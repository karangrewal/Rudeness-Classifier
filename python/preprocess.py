"""
This file runs OpenSMILE to extract MFCC/Prosodic features.

Usage: `python preprocess.py arg1`

Arguments:
0 -- MFCC features
1 -- Prosodic features

All labelled audio in `mfcc` and `prosody` is backed-up.
"""

import numpy as np
import os
import sys

# Path to OpenSMILE Config files
CONFIG = '/Users/karangrewal/opensmile-2.3.0/config/'

# Path to training data
PATH = '/Users/karangrewal/documents/developer/rudeness-classifier/conversations/'

# Subdirectores / Rudeness classes
CATEGORIES = ['insulting', 'interrupting', 'not_rude', 'shouting']

def clear_dir(path):
	"""
	Clear and delete a directory.
	"""
	if os.path.isdir(path):
		for item in os.listdir(path):
			os.remove(os.path.join(path, item))
		os.rmdir(path)

def mfcc_output_filename(fname, suffix, frames=False):
	if not frames:
		return os.path.join(PATH, 'mfcc_int', '%s%s' % (fname.split('.')[0], suffix))
	else:
		return os.path.join(PATH, 'mfcc', '%s%s' % (fname.split('.')[0], suffix))

def mfcc():

	clear_dir(os.path.join(PATH, 'mfcc_int'))
	os.mkdir(os.path.join(PATH, 'mfcc_int'))
	clear_dir(os.path.join(PATH, 'mfcc'))
	os.mkdir(os.path.join(PATH, 'mfcc'))

	# 1. Extract features from .wav files
	for category in CATEGORIES:
		examples = os.listdir(os.path.join(PATH, category))

		for f in examples:
			if f.endswith('.wav'):

				path_mfcc = mfcc_output_filename(f, '.csv')
				cmd = 'SMILExtract -C %s -I %s -csvoutput %s' % (os.path.join(CONFIG, 'MFCC12_0_D_A.conf'), os.path.join(PATH, category, f), path_mfcc)
				os.system(cmd)
	
				with open(path_mfcc) as g:
					data = np.genfromtxt(g, delimiter=';', skip_header=1)
					data = data[:,1:]

				np.savetxt(mfcc_output_filename(f, '.csv', True), data, delimiter=',')

	# Remove `mfcc-int` folder
	clear_dir(os.path.join(PATH, 'mfcc_int'))

def prosodic_output_filename(fname, suffix, type_feature, frames=False):
	if not frames:
		return os.path.join(PATH, 'prosody_int', '%s-%s%s' % (fname.split('.')[0], type_feature, suffix))
	else:
		return os.path.join(PATH, 'prosody', '%s%s' % (fname.split('.')[0], suffix))

def prosodic():

	clear_dir(os.path.join(PATH, 'prosody_int'))
	os.mkdir(os.path.join(PATH, 'prosody_int'))
	clear_dir(os.path.join(PATH, 'prosody'))
	os.mkdir(os.path.join(PATH, 'prosody'))

	# 1. Extract features from .wav files
	for category in CATEGORIES:
		examples = os.listdir(os.path.join(PATH, category))

		for f in examples:
			if f.endswith('.wav'):

				path_acf = prosodic_output_filename(f, 'acf', '.csv')
				cmd = 'SMILExtract -C %s -I %s -csvoutput %s' % (os.path.join(CONFIG, 'prosodyAcf.conf'), os.path.join(PATH, category, f), path_acf)
				os.system(cmd)
				path_shs = prosodic_output_filename(f, 'shs', '.csv')
				cmd = 'SMILExtract -C %s -I %s -csvoutput %s' % (os.path.join(CONFIG, 'prosodyShs.conf'), os.path.join(PATH, category, f), path_shs)
				os.system(cmd)

				# 2. Merge Acf and Shs prosodic features into single file
				with open(path_acf, 'r') as g:
					data1 = np.genfromtxt(g, delimiter=';', skip_header=1)
					data1 = data1[1:,1:]

				with open(path_shs, 'r') as g:
					data2 = np.genfromtxt(g, delimiter=';', skip_header=1)
					data2 = data2[:,2:]

				if data1.shape[0] != data2.shape[0]:
					min_len = min([data1.shape[0], data2.shape[0]])
					data1 = data1[:min_len,:]
					data2 = data2[:min_len,:]

				data = np.concatenate((data1, data2), axis=1)
				np.savetxt(prosodic_output_filename(f, '.csv', None, True), data, delimiter=',')

	# Remove `prosody-int` folder
	clear_dir(os.path.join(PATH, 'prosody_int'))

def preprocess(fname):
	"""
	** THIS UTILITY IS NO LONGER REQUIRED **

	Divide data into frames.
	"""
	with open(os.path.join(PATH, 'intermediate', fname), 'r') as f:
		data = np.genfromtxt(f, delimiter=';', skip_header=1)
		data = data[:,1:]

	samples = int((1. * data.shape[0] - K + 1) / W) + 1
	window = np.zeros(shape=(samples, data.shape[1]))

	for i in range(0, data.shape[0] - K + 1, W):
		window[i / W] = np.average(data[i:i+K,:], axis=0)

	np.round(window, decimals=4)
	np.savetxt(mfcc_output_filename(fname, '.csv', True), window, delimiter=',')

if __name__ == '__main__':
	if len(sys.argv) != 2:
		exit(0)
	arg1 = int(sys.argv[1])
	if arg1 == 0:
		mfcc()
	elif arg1 == 1:
		prosodic()