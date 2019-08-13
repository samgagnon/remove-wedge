"""
By Samuel Gagnon-Hartman

This script runs a validation test on a pre-existing neural network

"""

import torch as t
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import numpy.fft as fft
import pickle
import scipy.misc
import matplotlib.pyplot as plt
import sys

from feeder import DataFeeder
from model_def import *
from utils import *
from config import *

def verification(model, trainer=None, prefix=None, save_to=save_verification_to, verbose=False):

    """
    Creates a prediction set based on inputs upon which the network was not trained.

    model: model through which data is passed
    trainer: trainer that is used, if any
    prefix: prefix of files which are to be validated
    """
        
    feeder = DataFeeder(prefix=prefix)

    verification_inputs, verification_answers = feeder.get_batch(batch_size=batch_size)

    if verbose:
        print(np.shape(verification_inputs))
        print(np.shape(verification_answers))

    verification_inputs = convert_input_format(verification_inputs) # this is not normalized!
    verification_answers = normalize_map(convert_input_format(verification_answers))
    
    if verbose:
        print(np.shape(verification_inputs))
        print(np.shape(verification_answers))

    predictions = t.sigmoid(model.forward(verification_inputs))

    if trainer is None:
        loss = nn.BCELoss()
        loss_value = loss(predictions, verification_answers)
    else:
        loss_value = trainer.BCE_error(predictions, verification_answers, padding=border)

    return verification_inputs, predictions, verification_answers, loss_value.item()

def validation(validation_inputs, validation_answers, model, trainer):
    validation_predictions = model.forward(validation_inputs)
    validation_loss_value = trainer.BCE_error(validation_predictions, validation_answers, padding=border)
    return validation_loss_value.item()

def BCE_error(inputs, answers, padding=border):
    ans = answers[:, :, padding:-padding, padding:-padding]
    inp1 = F.logsigmoid(inputs[:, :, padding:-padding, padding:-padding])
    inp2 = F.logsigmoid(-inputs[:, :, padding:-padding, padding:-padding])
    err = - (ans*inp1 + (1-ans)*inp2)
    return t.mean(err)

if __name__ == "__main__":
    model = Net()
    model.load(model_filename)
    prefix = sys.argv[1]
    if prefix=='none':
        prefix = None
    feeder = DataFeeder(prefix=prefix)
    inputs, answers, val_inputs, val_answers = feeder.get_batch(batch_size=batch_size)
    val_inputs = normalize_map(convert_input_format(val_inputs))
    val_answers = normalize_map(convert_input_format(val_answers))
    validation_predictions = model.forward(val_inputs)
    validation_loss_value = BCE_error(validation_predictions, val_answers, padding=border)
    print("VALIDATION LOSS:", validation_loss_value.item())
    show = np.concatenate([val_inputs.cpu().data.numpy()[0,0],
                           validation_predictions.cpu().data.numpy()[0,0],
                           val_answers.cpu().data.numpy()[0,0]],1)
    plt.imsave('validation.png',show, cmap=plt.cm.hot)
    plt.imshow(show, cmap=plt.cm.hot)
    plt.show()
    val_answers_frac = t.mean((val_answers==val_answers.min()).float()).item()
    val_pred_frac = t.mean((validation_predictions-0.1<=validation_predictions.min()).float()).item()
    print("PREDICTIONS ZERO FRACTION*:", val_pred_frac)
    print("ANSWERS ZERO FRACTION:", val_answers_frac)
    print("Executed!")
    print("*fraction of data within 0.1 of the global minimum")