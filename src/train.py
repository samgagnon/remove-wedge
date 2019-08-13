"""
by Samuel Gagnon-Hartman, 2019

This script is used to train the models, essentially all parameters of the training you would want to ever change
are located in config.py

NEW FUNCTIONALITY:

1. now allows variable training rates, string types, and batch sizes for subsequent Gmu runs, 
the list of these rates is in config.py

2. now adds logging functionality, i.e. it appends to model.history at each training run, see model_def.py for more details

"""

import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.misc
import sys
import os

from verification import validation
from config import *
from model_def import *
from feeder import DataFeeder
from torchvision.transforms import Normalize
from utils import normalize_map, constant_bin_height_histogram, linearizer
from scipy.special import expit
from statistics import scatterplot
from scipy.stats import linregress, chisquare


# This class is used to train whole models, save them, etc. each instance of this class creates a new network
# either by initializing randomly or by loading a pre-trained one
class Trainer:

    # function which is called during the "trainer = Trainer(...)" line
    # it initializes the datafeeder, the network and the optimizer connected to the trainer
    def __init__(self, datafeeder, load_from=None):

        # we initialize the feeder object from which this instance intakes data
        self.feeder = datafeeder

        # if a starting file is specified, load it, this loads it in cpu
        if load_from is not None:
            self.net = load_model(load_from)
        else:
            # otherwise, detect the default network type defined in config.py
            if default_network_type == 'Net':
                self.net = Net()
            else:
                print('network_type must be Net')

        # if gpu is available, use it
        if t.cuda.is_available():
            self.net = self.net.cuda()

    # This defines the cross entropy function which we use to train the temp to string model, this is what the optimizer
    # will try to minimize. This is the cross-entropy of the network outputs with respect to the data
    # BCE stands for Binary Cross Entropy
    def BCE_error(self, inputs, answers, padding=border):

        # cmbedge string maps are not periodic,
        # We need to compute the error only on the pixels which do not exhibit edge effects
        # from the convolutions, a 10-pixel padding is more than enough for that
        ans = answers[:, :, padding:-padding, padding:-padding]
        size = ans.size()
        inp1 = F.logsigmoid(inputs[:, :, padding:-padding, padding:-padding])
        inp2 = t.log(t.ones(size, dtype=t.float32) - t.sigmoid(inputs[:, :, padding:-padding, padding:-padding]))
        err = - (ans*inp1 + (1-ans)*inp2)
        return t.mean(err)

    # This is the most used method, it will run num_steps of training steps while loading maps of given Gmu and noise
    # save_every refers to the number of training steps between model and image saves
    def train(self, num_steps, learning_rate=1e-3, noise=None, save_images=False, save_every=1, batch_size=1,
        save_models_to=models_directory, save_images_to=images_directory, log_file_location=log_file_location):

        # open the log file for writing
        log_file = open(log_file_location, 'w')
        loss_file_location = '..' + '\\' + 'loss_recorder' + '\\'
        loss_file_name = str(prefix) + '_loss.txt'
        validation_loss_file_name = str(prefix) + '_validation_loss.txt'
        os.system('mkdir' + loss_file_location)
        loss_file = open(loss_file_location + loss_file_name, 'w')
        validation_loss_file = open(loss_file_location + validation_loss_file_name, 'w')
        

        # define the optimizer to use, this is the an object which defines the algorithm which decides
        # how to increment the parameters given the gradient at each time-step
        # see the following paper for the exact algorithm: https://arxiv.org/abs/1412.6980
        # understanding how it works isn't that important, the only important thing is that it increments the
        # parameters roughly in the direction of the negative gradient at each time step
        # the learning_rate number determines the magnitude of weight updates, set it too large and the
        # training is unstable but fast, set it too small and the training is stable but too slow, it also risks
        # getting stuck too early, setting it to 0.001 works for most problems

        # The first argument is an iterable object (ie a list) of pytorch Variables which the optimizer will
        # optimize with respect to
        optimizer1 = optim.Adam(self.net.parameters(), lr=learning_rate)
        #optimizer2 = optim.Adam([self.net.a, self.net.b, self.net.c,
        #                          self.net.d, self.net.e], lr=learning_rate*1000)

        # add training run to the network object
        self.net.add_training_run_to_history(n_iterations=0,
                                             noise_std=0 if noise is None else noise,
                                             batch_size=batch_size,
                                             learning_rate=learning_rate,
                                             optimization_goal='BCE_error')

        start_index = 0
        for i in range(1, num_steps+1):

            print('Epoch ' + str(i))
            print('START INDEX:', start_index)
            # this gets the inputs and answer maps from the feeder, set random_indices=False if you don't want them
            # randomly sampled, but want the same every time
            inputs, answers, validation_inputs, validation_answers = self.feeder.get_batch(batch_size=batch_size,
                                                                                           start_index=start_index,
                                                                                           noise=noise, 
                                                                                           random_indices=False, 
                                                                                           gpu_flag=False)

            # zero the gradient buffers, if this is not done, the gradients will be wrongly computed
            optimizer1.zero_grad()
            #optimizer2.zero_grad()
            inputs = normalize_map(convert_input_format(inputs))
            answers = normalize_map(convert_input_format(answers))
            validation_inputs = normalize_map(convert_input_format(validation_inputs))
            validation_answers = normalize_map(convert_input_format(validation_answers))
            #print("Verification of input and answer Normalization")
            #print(inputs.max(), answers.max(), inputs.min(), answers.min())
            # inputs = inputs/inputs.std()
            output = self.net.forward(inputs)  # evaluate the network on the inputs
            prediction = linearizer(output, self.net.a, self.net.b, self.net.c,
                                    self.net.d, self.net.e, on=False) # applies linearizer to outputs cell-wize
            #loss = self.BCE_error(prediction, answers, padding=25)  # evaluate the loss on the output and answers
            loss_value = self.BCE_error(prediction, answers, padding=border)
            print("LOSS:", loss_value.item())
            loss_value.backward(retain_graph=False) # this line computes the derivatives with respect to the networks
            optimizer1.step()  # this updates the parameters in self.net based on the optimizer equations
            
            # computes loss on predictions made on data upon which the network was not trained
            # if validation loss increases with time, then the trainer immediately proceeds to the next learning rate
            validation_loss = validation(validation_inputs, validation_answers, model=self.net, trainer=self)
            print("VALIDATION LOSS:", validation_loss)
                
            self.net.add_iteration_to_current_training_run(1)  # for logging purposes
            if start_index+batch_size<9*batch_size:
                start_index+=batch_size
            else:
                start_index = 0
            
            # writes loss to file
            if learning_rate is learning_rates_list[0] and i is 1:
                validation_loss_file.write(str(validation_loss))
                loss_file.write(str(loss_value.item()))
            else:
                validation_loss_file.write(',' + str(validation_loss))
                loss_file.write(',' + str(loss_value.item()))

            # If the time step is a multiple of save_every, save the model
            if i % save_every == 0:
                self.net = self.net.cpu()  # send to cpu to save the weights on cpu
                self.net.save(save_models_to + '\\' + 'prediction_model.pth')
                # send back
                if t.cuda.is_available():
                    self.net = self.net.cuda()


            # if the time step is a multiple of save_every, save an image showing the input,
            # the output and the answer side-by-side
            if i % save_every == 0 and save_images:
                #fig, mapped_plot = scatterplot(answers.cpu().data.numpy()[0, 0], prediction.cpu().data.numpy()[0, 0])
                show_in = inputs.cpu().data.numpy()[0, 0]
                show_out = prediction.cpu().data.numpy()[0, 0]
                show_ans = answers.cpu().data.numpy()[0, 0]
                plot  = scatterplot(show_ans, show_out)

                show = np.concatenate([show_in, show_out, show_ans], axis=1)
                
                plt.imsave(save_images_to + '\\' + 'i_' + str(i) + '_error_' + str(loss_value.item()) + '.png',
                    show, cmap=plt.cm.hot)

                plot.savefig(save_images_to + '\\' + 'i_' + str(i) + 'scatterplot.png')

            log_file.write("time step: " + str(i) + " error: " + str(loss_value.item()) + '\n')
        loss_file.close()
        validation_loss_file.close()
        log_file.close()


def train_main():
    # initialize the feeder and trainer objects
    feeder = DataFeeder(prefix=prefix, verbose=True)
    trainer = Trainer(feeder, load_from=start_training_from_this_model_filename)

    # creates folders in which we store models and images
    os.system('mkdir ' + models_directory)
    os.system('mkdir ' + images_directory)

    # start the training loops, this for loop basically iterates over the 2 lists inside the zip(...) at the same time
    for learning_rate, num_steps_per_train_run in zip(learning_rates_list,
                                                    num_steps_per_train_run_list):

        os.system('mkdir' + models_directory + '\\' + 'learn_rate_' + str(learning_rate))
        os.system('mkdir' + images_directory + '\\' + 'learn_rate_' + str(learning_rate))
        
        for noise in [None]:
            
            # let's try iterating over learn rate
            save_models_to = models_directory + '\\' + 'learn_rate_' + str(learning_rate)
            save_images_to = images_directory + '\\' + 'learn_rate_' + str(learning_rate)

            os.system('mkdir' + save_models_to)
            os.system('mkdir' + save_images_to)

            # call the trainer
            trainer.train(num_steps=num_steps_per_train_run,
                          learning_rate=learning_rate,
                          noise=noise, save_images=True, save_every=save_model_and_images_every,
                          batch_size=batch_size,
                          save_models_to=save_models_to, save_images_to=save_images_to,
                          log_file_location=save_models_to + '\\' + 'log_file' + '.txt')


if __name__ == '__main__':
    global prefix
    prefix = sys.argv[1]
    if prefix == 'none':
        prefix = None
    train_main()