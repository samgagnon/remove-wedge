'''
created by: Samuel Gagnon-Hartman
'''
import torch as t
from feeder import *
from model_def import *
from config import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from verification import verification
from utils import normalize_map

def BCE_error(inputs, answers, padding=20):

    # cmbedge string maps are not periodic,
    # We need to compute the error only on the pixels which do not exhibit edge effects
    # from the convolutions, a 10-pixel padding is more than enough for that
    # hahaha I bet Razvan knew what he was doing! I don't. Let's just apply this to every
    # pixel on every map.

    ans = answers[padding:-padding, padding:-padding]
    #ans = (answers[:, :, padding:-padding, padding:-padding]).float()
    inp1 = F.logsigmoid(inputs[padding:-padding, padding:-padding])
    inp2 = F.logsigmoid(-inputs[padding:-padding, padding:-padding])
    err = - (ans*inp1 + (1-ans)*inp2)

    return t.mean(err)

# loads binary data and stores in numpy array
def load_binary_data(filename, dtype=np.float32): 
	"""
	We assume that the data was written 
	with write_binary_data() (little endian). 
	""" 
	f = open(filename, "rb")
	data = f.read()
	f.close()
	_data = np.frombuffer(data, dtype)
	if sys.byteorder == 'big':
		_data = _data.byteswap()
	return _data

def shape_data(dimension, data):
	data.shape = (dimension,dimension,dimension)
	data = data.reshape((dimension,dimension,dimension), order = 'F')
	return data

def cube_loss():
    '''
    This method examines the cross-entropy loss between the first slice of a cube and all the maps in
    succeeding indices for all three dimensions of the cube.
    '''
    file_exists = False
    target = 'delta_T'
    target_name_length = len(target)
    for directory in os.listdir(training_data_folder):
        if basename(directory)[0] == 'z' and basename(directory)[2] != '1' and basename(directory)[2] != '7':
            file_folder = training_data_folder + directory
            for folder in os.listdir(file_folder):
                if basename(folder)[0:3] == 'Run':
                    file_subfolder = file_folder + '\\' + folder
                    for filename in os.listdir(file_subfolder):
                        if basename(filename)[0:target_name_length] == target:
                            file_exists = True
                            data = load_binary_data(file_subfolder + '\\' + filename)
                            DIM = int("" + filename.split("_")[-2])
                            data = shape_data(DIM, data)
                            break
    data = (data - np.mean(data))/np.std(data)
    first_slice = data[0]
    loss_list = [BCE_error(t.tensor(first_slice), t.tensor(data[i,:,:])).item() for i in range(len(data[0,0]))]
    print("LOSS BETWEEN MAP AND ITSELF:", BCE_error(t.tensor(first_slice),t.tensor(first_slice)).item())
    plt.plot(loss_list)
    plt.show()

def scatterplot(prediction, truth):
    """
    Generates a scatterplot of predicted versus true map values, given the two maps
    -prediction: the predicted map
    -truth: the true map
    """
    figure = plt.figure()
    plt.plot(truth, prediction, "ob", alpha=0.05)
    plt.xlabel("Truth Values")
    plt.ylabel("Precicted Values")
    return figure

if __name__ == "__main__":
    cube_loss()