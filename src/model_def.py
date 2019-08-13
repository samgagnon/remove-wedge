import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

from config import *
from utils import normalize_map
from heapq import nlargest
from torch.autograd import Variable
from feeder import DataFeeder
from random import uniform

# converts input from the dataFeeder method get_batch to shape [B, 1, N, N]
def convert_input_format(x):
    '''
    Converts input from list of numpy arrays into Variable object usable by PyTorch
    '''
    x = np.array(x)
    converted_input = t.FloatTensor(x).unsqueeze(1)
    return Variable(converted_input)

# class defining a convolution operation with the assumption of periodic boundary conditions
# st0len from Razvan's code
# Razvan used this to hide the fact that his training data had periodic boundary conditions,
# if this is not the case for us, don't use it
class RepeatingConv(nn.Module):

    def __init__(self, in_channels, out_channels, conv_size):
        super(RepeatingConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, conv_size, 1, 0)
        #self.conv.weight.data /= 1
        self.conv_size = conv_size

    def pad_repeating(self, x, pad_size):
        s2 = x.size(2)
        s3 = x.size(3)
        x1 = t.cat([x[:, :, s2 - pad_size: s2], x, x[:, :, 0:pad_size]], 2)
        x2 = t.cat([x1[:, :, :, s3 - pad_size: s3], x1, x1[:, :, :, 0:pad_size]], 3)
        return x2

    def forward(self, x):
        if self.conv_size == 1:
            return self.conv(x)
        else:
            x = self.pad_repeating(x, int((self.conv_size-1)/2))
            x = self.conv(x)
            return x

# mainly provides uniform saving/loading and history logging facilities
# the model.history is an array containing stuff that happened to the model in chronological order
# Our network Net and other future networks descend from it
# st0len from Razvan's code
class CosmicDawnModel(nn.Module):

    def __init__(self):
        super(CosmicDawnModel, self).__init__()
        self.history = []

    # used to log a training run into model.history
    def add_training_run_to_history(self, n_iterations, noise_std, batch_size, learning_rate, optimization_goal):
        # add any important features of the training run which should be recorded

        training_run = {'n_iterations': n_iterations,
                        'noise_std:' : noise_std,
                        'batch_size:': batch_size,
                        'learning_rate:': learning_rate,
                        'optimization_goal': optimization_goal}

        self.history.append(training_run)

    # used to add an iteration to the latest training run
    def add_iteration_to_current_training_run(self, n_iterations_to_add = 1):
        self.history[-1]['n_iterations'] += n_iterations_to_add

    # saves the current weights to filename
    def save(self, filename):
        to_save = [self.network_description, self.history, self.state_dict()]
        pickle.dump(to_save, open(filename, 'wb'))

    # loads the weights from filename into the current net
    def load(self, filename):
        self.network_description, self.history, state_dict = pickle.load(open(filename, 'rb'))
        self.load_state_dict(state_dict)

# function for generalized model loading, if you define more models you need to add them here
# each model has a model.network description variable which contains a decription of which it is, we use that variable
# to detect which model it is
def load_model(filename):

    network_description, history, state_dict = pickle.load(open(filename, 'rb'))

    if network_description.split('_')[0] == 'Net':
        model = Net()
    else:
        print('sorry, network type is not recognized')
        return None

    model.history = history
    model.load_state_dict(state_dict)

    return model

# model defining a primitive network used in the first paper
class Net(CosmicDawnModel):
    def __init__(self):
        super(Net, self).__init__()
        conv_size = 3
        channels  = 12
        self.network_description = 'Net'
        #self.lin_params = t.tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0])).float().requires_grad_()
        self.a = t.tensor(np.array([1])).float().requires_grad_()
        self.b = t.tensor(np.array([1])).float().requires_grad_()
        self.c = t.tensor(np.array([1.01])).float().requires_grad_()
        self.d = t.tensor(np.array([1])).float().requires_grad_()
        self.e = t.tensor(np.array([1])).float().requires_grad_()

        # each of these self.conv is an instance of the RepeatingConv class
        
        self.conv1 = RepeatingConv(1, channels, conv_size)
        self.conv2 = RepeatingConv(channels, channels, conv_size)
        self.conv3 = RepeatingConv(channels, channels, conv_size)
        self.conv4 = RepeatingConv(channels, channels, conv_size)
        self.conv7 = nn.Conv2d(channels, 1, 1, 1, 0)
        # self.drop1 = nn.Dropout2d(dropout_rate)
        # self.drop2 = nn.Dropout2d(dropout_rate)

        # the last one is named with 7 instead of 5 because of backward compatibility with older versions,
        # changing it will cause ./starting_model.pth to no longer be loadable

    # this is the function where the network operations actually take place
    def forward(self, x):
        y = t.tanh(self.conv1.forward(x))
        # y = self.drop1(y)
        y = t.tanh(self.conv2.forward(y))
        # y = self.drop2(y)
        y = t.tanh(self.conv3.forward(y))
        #y = self.drop3(y)
        y = t.tanh(self.conv4.forward(y))
        #y = self.drop4(y)
        y = self.conv7(y)

        return y

class LineNet(CosmicDawnModel):
    def __init__(self):
        super(LineNet, self).__init__()
        conv_size = 3
        channels = 1
        self.network_description = 'Net'
        #self.lin_params = t.tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0])).float().requires_grad_()
        self.a = t.tensor(np.array([1])).float().requires_grad_()
        self.b = t.tensor(np.array([1])).float().requires_grad_()
        self.c = t.tensor(np.array([1.01])).float().requires_grad_()
        self.d = t.tensor(np.array([1])).float().requires_grad_()
        self.e = t.tensor(np.array([1])).float().requires_grad_()

        # each of these self.conv is an instance of the RepeatingConv class
        
        self.conv1 = RepeatingConv(1, channels, conv_size)

    # this is the function where the network operations actually take place
    def forward(self, x):
        x = self.conv1.forward(x)
        return x

if __name__ == '__main__':
    print("Debugging Model_def")
    # instantiate a model, this is not a trained network and will have random weights
    model = Net()

    # if we want to transfer the model from cpu to gpu or from gpu to cpu, we use the following lines, we need a
    # model which lives in gpu to evaluate it on maps in gpu memory
    if t.cuda.is_available():  # verifies if gpu is available on the current machine
        model = model.cuda()  # transfers the model to gpu memory
        model = model.cpu()  # transfers it back to cpu memory

    # instantiate a feeder
    feeder = DataFeeder(prefix='sweep-10',verbose=True)

    # get a data batch from the feeder
    inputs, answers, val_in, val_ans = feeder.get_batch()

    print(np.shape(inputs))

    inputs = convert_input_format(inputs)
    answers = convert_input_format(answers)
    
    print(np.shape(inputs))