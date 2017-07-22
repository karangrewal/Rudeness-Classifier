"""
Replicating Experiment in: https://infoscience.epfl.ch/record/203865/files/Yella_SLT_2014.pdf
"""

from lasagne.layers import DenseLayer, InputLayer
from lasagne.layers import get_all_params, get_output, get_all_layers
from lasagne.nonlinearities import leaky_rectify, linear, rectify, sigmoid, softmax, LeakyRectify
from lasagne.objectives import binary_crossentropy
from lasagne.regularization import l2, regularize_network_params
from lasagne.updates import adam, rmsprop, sgd
import numpy as np
import os
import sys
from theano import function
import theano.tensor as T

ALPHA = 2e-3
BATCH_SIZE = 32
COMPARISON_FRAMES = 100
DATA_PATH = '/Users/karangrewal/documents/developer/rudeness-classifier/conversations/mfcc-2'
DATA_SIZE = 14000
DIM_H1 = 300
DIM_H2 = 250
EPOCHS = 35
L2_REGULARIZATION = True
LR = 0.01
PROB_SELECT_AS_TEST = 0.1
NUM_FEATURES = 13
MFCC_NUM = 1
OUT_PATH = ''
WEIGHT_CONSTRAINT = True
WINDOW_SIZE = 4

def get_data(n=DATA_SIZE, same=7000, diff=7000):
    ''' Files have 42 cols '''
    if same + diff != n:
        exit(0)
    print('GATHERING DATA ...')
    x_y_train, x_y_test = None, None
    example_files = os.listdir(DATA_PATH)

    def random():
        return np.random.uniform() < PROB_SELECT_AS_TEST

    # Read files
    for f in example_files:
        if f.endswith('.csv'):

            file_data = np.genfromtxt(os.path.join(DATA_PATH, f), delimiter=',', skip_header=0)
            assert file_data.shape[1] == 42
            
            # Choose f as test file with probability `PROB_SELECT_AS_TEST`
            if random():
                print('Test Case: {}'.format(f))
                if x_y_test is not None:
                    x_y_test = np.concatenate((x_y_test, file_data), axis=0)
                else:
                    x_y_test = file_data
            else:
                if x_y_train is not None:
                    x_y_train = np.concatenate((x_y_train, file_data), axis=0)
                else:
                    x_y_train = file_data
    
    assert x_y_test.shape[0] > 0
    x_y_train, x_y_test = np.float32(x_y_train), np.float32(x_y_test)
    x_y_train = x_y_train[x_y_train[:,41]!=0,:]
    x_y_test = x_y_test[x_y_test[:,41]!=0,:]

    # Select data
    if MFCC_NUM == 1:
        x_y_train = x_y_train[:,np.r_[1:14, 41]]
        x_y_test = x_y_test[:,np.r_[1:14, 41]]
    elif MFCC_NUM == 2:
        x_y_train = x_y_train[:,np.r_[14:27, 41]]
        x_y_test = x_y_test[:,np.r_[14:27, 41]]
    elif MFCC_NUM == 3:
        x_y_train = x_y_train[:,np.r_[27:40, 41]]
        x_y_test = x_y_test[:,np.r_[27:40, 41]]

    # Randomly select 2 examples and create data
    n_same, n_diff = 0, 0
    train_data = np.zeros(shape=(0, 2*WINDOW_SIZE*NUM_FEATURES+1))
    test_data = np.zeros(shape=(0, 2*WINDOW_SIZE*NUM_FEATURES+1))
    train_speakers, test_speakers = list(), list()
    
    # Generate Training Data
    while n_same + n_diff < n:
        a, b = np.random.randint(0, x_y_train.shape[0]-WINDOW_SIZE), np.random.randint(0, x_y_train.shape[0]-WINDOW_SIZE)
        if x_y_train[a,-1] != x_y_train[a+WINDOW_SIZE,-1] or x_y_train[b,-1] != x_y_train[b+WINDOW_SIZE,-1]:
            continue
        
        x1 = np.concatenate((x_y_train[a:a+WINDOW_SIZE,:-1].flatten(), x_y_train[a:a+1,-1]))
        x2 = np.concatenate((x_y_train[b:b+WINDOW_SIZE,:-1].flatten(), x_y_train[b:b+1,-1]))

        if x1[-1] == x2[-1] and n_same < same:
            n_same += 1
            train_speakers.append(x1[-1])
            train_speakers.append(x2[-1])
            train_data = np.append(train_data, np.concatenate((x1[:-1], x2)).reshape(-1, 2*WINDOW_SIZE*NUM_FEATURES+1), axis=0)
            train_data[-1,-1] = 1
        elif x1[-1] != x2[-1] and n_diff < diff:
            n_diff += 1
            train_speakers.append(x1[-1])
            train_speakers.append(x2[-1])
            train_data = np.append(train_data, np.concatenate((x1[:-1], x2)).reshape(-1, 2*WINDOW_SIZE*NUM_FEATURES+1), axis=0)
            train_data[-1,-1] = 0

        if n_same + n_diff % 2000 == 0:
            print(n_same + n_diff)
    
    # Generate Test Data
    # Pairs consist of MFCC pairs which are 0.1s apart
    for i in range(x_y_test.shape[0] - COMPARISON_FRAMES - WINDOW_SIZE + 1):
        if np.unique(x_y_test[i:i+WINDOW_SIZE,-1]).shape[0] != 1 or np.unique(x_y_test[i+COMPARISON_FRAMES:i+COMPARISON_FRAMES+WINDOW_SIZE,-1]).shape[0] != 1:
            continue

        x1 = np.concatenate((x_y_test[i:i+WINDOW_SIZE,:-1].flatten(), x_y_test[i:i+1,-1]))
        x2 = np.concatenate((x_y_test[i+COMPARISON_FRAMES:i+COMPARISON_FRAMES+WINDOW_SIZE,:-1].flatten(), x_y_test[i+COMPARISON_FRAMES:i+101,-1]))
        test_data = np.append(test_data, np.concatenate((x1[:-1], x2)).reshape(-1, 2*WINDOW_SIZE*NUM_FEATURES+1), axis=0)
        test_speakers.append(x1[-1])
        test_speakers.append(x2[-1])

        if x1[-1] == x2[-1]:
            test_data[-1,-1] = 1
        elif x1[-1] != x2[-1]:
            test_data[-1,-1] = 0

    # Print Speakers
    print('Training Data Speakers: %s' % np.sort(np.unique(np.array(train_speakers))))
    print('Test Data Speakers: %s' % np.sort(np.unique(np.array(test_speakers))))

    # Split training and test data
    return train_data, test_data

def network(input_var=None):
    ann = InputLayer(shape=(None, 2*WINDOW_SIZE*NUM_FEATURES), input_var=input_var)
    ann = DenseLayer(ann, DIM_H1, nonlinearity=sigmoid)
    ann = DenseLayer(ann, DIM_H2, nonlinearity=sigmoid)
    ann = DenseLayer(ann, 2, nonlinearity=softmax)
    return ann

if __name__ == '__main__':

    x = T.fmatrix()
    t = T.fvector()
    ann = network(x)

    prediction = get_output(ann)[:,1]
    predict = function([x], outputs=prediction)

    loss = binary_crossentropy(prediction, t).mean()

    # L2 regularization
    if L2_REGULARIZATION:
        l2_penalty = ALPHA * regularize_network_params(ann, l2)
        loss += l2_penalty.mean()

    updates = sgd(loss_or_grads=loss, params=get_all_params(ann, trainable=True), learning_rate=LR)
    train = function([x, t], outputs=loss, updates=updates, allow_input_downcast=True, mode='FAST_COMPILE')

    # Load data
    train_data, test_data = get_data()
    train_data, test_data = np.float32(train_data), np.float32(test_data)

    # Standardize features
    # train_data[:,:-1] = (train_data[:,:-1] - np.mean(train_data[:,:-1], axis=0)) / np.std(train_data[:,:-1], axis=0)
    # test_data[:,:-1] = (test_data[:,:-1] - np.mean(train_data[:,:-1], axis=0)) / np.std(train_data[:,:-1], axis=0)

    # Scale between -1 and 1
    train_data[:,:-1] = train_data[:,:-1] / (np.max(train_data[:,:-1], axis=0) - np.min(train_data[:,:-1], axis=0))
    test_data[:,:-1] = test_data[:,:-1] / (np.max(train_data[:,:-1], axis=0) - np.min(train_data[:,:-1], axis=0))

    print('STARTING TRAINING ...')
    for epoch in range(EPOCHS):
        
        # Reorder data
        # loss = np.zeros(shape=(DATA_SIZE/BATCH_SIZE))
        loss = np.zeros(shape=(0))
        for i in range(0, DATA_SIZE, BATCH_SIZE):
            
            x_i = train_data[i:i+BATCH_SIZE,:-1]
            t_i = train_data[i:i+BATCH_SIZE,-1]
            loss = np.append(loss, np.array([train(x_i, t_i)]))
            # print(train(x_i, t_i))

            # Constrain weights
            params = get_all_params(ann)
            if WEIGHT_CONSTRAINT:
                w = params[0].get_value()
                w[:WINDOW_SIZE*NUM_FEATURES, DIM_H1/2:].fill(0.)
                w[WINDOW_SIZE*NUM_FEATURES:, :DIM_H1/2].fill(0.)
                params[0].set_value(w)

        print('Epoch {}/{}: Loss={}'.format(epoch+1, EPOCHS, np.average(loss)))
        if (epoch+1) % 5 == 0:
            # print('\tAvg. W1: {}'.format(np.average(params[0].get_value())))
            # print('\tAvg. b1: {}'.format(np.average(params[1].get_value())))
            # print('\tAvg. W2: {}'.format(np.average(params[2].get_value())))
            # print('\tAvg. b2: {}'.format(np.average(params[3].get_value())))
            print('\tAvg. W1: {}'.format(np.average(np.absolute(params[0].get_value()))))
            print('\tAvg. b1: {}'.format(np.average(np.absolute(params[1].get_value()))))
            print('\tAvg. W2: {}'.format(np.average(np.absolute(params[2].get_value()))))
            print('\tAvg. b2: {}'.format(np.average(np.absolute(params[3].get_value()))))

        # Record weights
        # np.savez(os.path.join(OUT_PATH, 'w_%d' % (epoch+1)), params[0].get_value())

    print('MAKING PREDICTIONS ...')

    # Make Predictions
    Y = None
    for i in range(0, test_data.shape[0], BATCH_SIZE):
        x_i = test_data[i:i+BATCH_SIZE,:-1]
        y_i = predict(x_i)
        y_i = np.round(y_i)
        t_i = test_data[i:i+BATCH_SIZE,-1]

        assert y_i.shape[0] == t_i.shape[0]
        y_i = y_i.reshape(-1, 1)
        t_i = t_i.reshape(-1, 1)

        if Y is None:
            Y = np.concatenate((y_i, t_i), axis=1)
        else:
            Y = np.concatenate((Y, np.concatenate((y_i, t_i), axis=1)), axis=0)

    print('Test Samples 0: {}\tText Samples 1: {}'.format(Y[Y[:,1]==0,:].shape[0], Y[Y[:,1]==1,:].shape[0]))
    print('Guessed 0: {}\tGuessed 1: {}'.format(Y[Y[:,0]==0,:].shape[0], Y[Y[:,0]==1,:].shape[0]))

    acc = np.sum(Y[:,0] == Y[:,1]) * 100. / Y.shape[0]
    print('Test Accuracy: {}%'.format(acc))

    # Save results
    np.savetxt(os.path.join(OUT_PATH, 'ann_results_%s.txt' % sys.argv[1]), Y, delimiter=',')
