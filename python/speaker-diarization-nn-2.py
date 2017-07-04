from lasagne.layers import DenseLayer, InputLayer
from lasagne.layers import get_all_params, get_output, get_all_layers
from lasagne.nonlinearities import leaky_rectify, linear, rectify, sigmoid, softmax, LeakyRectify
from lasagne.objectives import binary_crossentropy
from lasagne.updates import sgd
import numpy as np
import os
from theano import function
import theano.tensor as T

BATCH_SIZE = 32
DATA_SIZE = 10000
DIM_H1 = 60
EPOCHS = 50
NUM_FEATURES = 13
MFCC_NUM = 1
DATA_PATH = '/Users/karangrewal/documents/developer/rudeness-classifier/conversations/mfcc-2'
OUT_PATH = ''


def get_data(n=DATA_SIZE, same=5000, diff=5000, factor=1.2):
    ''' Files have 42 cols '''
    print('GATHERING DATA ...')
    if same + diff != n:
        exit(0)
    x_y_train = None
    example_files = os.listdir(DATA_PATH)

    # Read files
    for f in example_files:
        if f.endswith('.csv'):

            file_data = np.genfromtxt(os.path.join(DATA_PATH, f), delimiter=',', skip_header=0)
            assert file_data.shape[1] == 42

            if x_y_train is not None:
                x_y_train = np.concatenate((x_y_train, file_data), axis=0)
            else:
                x_y_train = file_data
    x_y_train = np.float32(x_y_train)
    x_y_train = x_y_train[x_y_train[:,41]!=0,:]

    # Select data
    if MFCC_NUM == 1:
        x_y_train = x_y_train[:,np.r_[1:14, 41]]
    elif MFCC_NUM == 2:
        x_y_train = x_y_train[:,np.r_[14:27, 41]]
    elif MFCC_NUM == 3:
        x_y_train = x_y_train[:,np.r_[27:40, 41]]

    # Randomly select 2 examples and create data
    n_same, n_diff = 0, 0
    data = np.zeros(shape=(0, 2*NUM_FEATURES+1))
    
    # Generate samples
    while n_same + n_diff < n * factor:
        a, b = np.random.randint(0, x_y_train.shape[0]), np.random.randint(0, x_y_train.shape[0])
        x1, x2 = x_y_train[a].reshape(x_y_train.shape[1]), x_y_train[b,:].reshape(x_y_train.shape[1])

        if x1[-1] == x2[-1] and n_same < same * factor:
            n_same += 1
            data = np.append(data, np.concatenate((x1[:-1], x2)).reshape(-1, 2*NUM_FEATURES+1), axis=0)
            data[-1,-1] = 1
        elif x1[-1] != x2[-1] and n_diff < diff * factor:
            n_diff += 1
            data = np.append(data, np.concatenate((x1[:-1], x2)).reshape(-1, 2*NUM_FEATURES+1), axis=0)
            data[-1,-1] = 0

    # Split training and test data
    np.random.shuffle(data)
    return data[:DATA_SIZE,:], data[DATA_SIZE:,:]

def network(input_var=None):
    ann = InputLayer(shape=(None, 2*NUM_FEATURES), input_var=input_var)
    ann = DenseLayer(ann, DIM_H1)
    ann = DenseLayer(ann, 48)
    ann = DenseLayer(ann, 2, nonlinearity=softmax)
    return ann

if __name__ == '__main__':

    x = T.fmatrix()
    t = T.fvector()
    ann = network(x)

    prediction = T.argmax(get_output(ann), axis=1)
    predict = function([x], outputs=prediction)

    loss = binary_crossentropy(prediction, t).mean()
    updates = sgd(loss_or_grads=loss, params=get_all_params(ann, trainable=True), learning_rate=0.01)

    # test_fn = function([x, t], [test_loss, test_acc])
    # test_acc = T.mean(T.eq(T.argmax(t, axis=1), t),dtype=theano.config.floatX)

    train = function([x, t], outputs=loss, updates=updates, allow_input_downcast=True)

    # Load data
    train_data, test_data = get_data()
    train_data, test_data = np.float32(train_data), np.float32(test_data)

    print('STARTING TRAINING ...')
    for epoch in range(EPOCHS):

        # Reorder data
        np.random.shuffle(train_data)
        k = 0
        for i in range(0, DATA_SIZE, BATCH_SIZE):
            
            x_i = train_data[i:i+BATCH_SIZE,:-1]
            t_i = train_data[i:i+BATCH_SIZE,-1]
            loss = train(x_i, t_i)

            # Constrain weights
            params = get_all_params(ann)
            w = params[0].get_value()
            w[:NUM_FEATURES, DIM_H1/2:].fill(0.)
            w[NUM_FEATURES:, :DIM_H1/2].fill(0.)
            params[0].set_value(w)

        # Record weights
        # np.savez(os.path.join(OUT_PATH, 'w_%d' % (epoch+1)), params[0].get_value())

    print('MAKING PREDICTIONS ...')

    # Make Predictions
    Y = None
    for i in range(0, test_data.shape[0], BATCH_SIZE):
        x_i = test_data[i:i+BATCH_SIZE,:-1]
        y_i = predict(x_i)
        t_i = test_data[i:i+BATCH_SIZE,-1]

        assert y_i.shape[0] == t_i.shape[0]
        y_i = y_i.reshape(-1, 1)
        t_i = t_i.reshape(-1, 1)

        if Y is None:
            Y = np.concatenate((y_i, t_i), axis=1)
        else:
            Y = np.concatenate((Y, np.concatenate((y_i, t_i), axis=1)), axis=0)

    acc = np.sum(Y[:,0] == Y[:,1]) * 100. / Y.shape[0]
    print('Accuracy: {}%'.format(acc))

    # Save results
    np.savetxt(os.path.join(OUT_PATH, 'ann_results.txt'), Y, delimiter=',')
