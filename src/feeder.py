"""
by Samuel Gagnon-Hartman, 2019

This script defines a DataFeeder class which loads 21cm signal maps and provides 
a method for sampling from them. This is mainly used by train.py to get the training data.

"""
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import sys
from os.path import basename

import torch as t
from torch.autograd import Variable
from scipy.misc import imshow
from random import shuffle
from config import *
from heapq import nlargest
from heapq import nsmallest
from utils import *


# This class handles the feeding of data, i.e. it handles loading the relevant files, calling the scripts which
# produce the numpy versions of the maps, and combining the data into a format appropriate for feeding
# to the network for training.
class DataFeeder:
	"""
	The DataFeeder object is used primarily in train.py to load numpy
	data and train the neural network.
	"""
	
	# loads binary data and stores in numpy array
	def load_binary_data(self, filename, dtype=np.float32): 
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

	def shape_data(self, dimension, data):
		data.shape = (dimension,dimension,dimension)
		data = data.reshape((dimension,dimension,dimension), order = 'F')
		return data
	
	# reduces size of data stored in the DataFeeder
	def slice_selector(self, x):
		'''
    	Converts input cube of type np.ndarray into a list of numpy arrays representing
		necessary (uncorrelated) cube slices
    	'''
    	# the number of maps is four times the number of map cubes passed into the method
    	# this is at the current sampling rate of four maps per cube
		cube_slices1 = [x[i,:,:] for i in range(0,199,50)]
		cube_slices2 = [x[:,i,:] for i in range(0,199,50)]
		cube_slices3 = [x[:,:,i] for i in range(0,199,50)]
		cube_slices = cube_slices1 + cube_slices2 + cube_slices3
		return cube_slices

	# initialization routine used when feeder = DataFeeder()
	def __init__(self, prefix=None, training_data_folder=training_data_folder, verbose=False, randomize=True):
		if verbose:
			print("Starting to load files into DataFeeder...")
		
		self.xH = []
		self.T = []
		file_exists = False
		xH_maps_counter = 0
		delta_T_maps_counter = 0

		target = 'fill-10'
		target_name_length = len(target)

		if prefix is None:
			train_filename = 'delta_T'
			train_name_length = len(train_filename)
		else:
			train_filename = prefix + '-delta_T'
			train_name_length = len(prefix) + 8

		for directory in os.listdir(training_data_folder):
			if basename(directory)[0] == 'z' and basename(directory)[2] != '1' and basename(directory)[2] != '7':
				if verbose:
					print(directory)
				file_folder = training_data_folder + directory
				for folder in os.listdir(file_folder):
					if basename(folder)[0:3] == 'Run':
						if verbose:
							print(folder)
						file_subfolder = file_folder + '\\' + folder
						for filename in os.listdir(file_subfolder):
							# loads neutral hydrogen maps
							if basename(filename)[0:target_name_length] == target and xH_maps_counter<xH_maps_max:
								if verbose:
									print(filename)
								file_exists = True # this is wasteful, find a better way to do this
								data = self.load_binary_data(file_subfolder + '\\' + filename)
								DIM = int("" + filename.split("_")[-2])
								data = self.shape_data(DIM,data)
								data = self.slice_selector(data)
								self.xH = self.xH + data
								xH_maps_counter+=1
							# loads hydrogen temperature maps
							elif basename(filename)[0:train_name_length] == train_filename and delta_T_maps_counter<delta_T_maps_max:
								if verbose:
									print(filename)
								file_exists = True # wasteful and dumb, fix this
								data = self.load_binary_data(file_subfolder + '\\' + filename)
								DIM = int("" + filename.split("_")[-2])
								data = self.shape_data(DIM,data)
								data = self.slice_selector(data)
								self.T = self.T + data
								delta_T_maps_counter+=1

		if not file_exists:
			raise Exception("The necessary files are either improperly named or nonexistent.")
		
		self.map_shape = np.shape(self.xH)

		# split the map pool into training and validation sets
		self.split_index = int(3*self.map_shape[0]/4)
		self.train_T = self.T[0:self.split_index]
		self.train_xH = self.xH[0:self.split_index]
		self.val_T = self.T[self.split_index:self.map_shape[0]]
		self.val_xH = self.xH[self.split_index:self.map_shape[0]]

	# function which translates maps by (t_x, t_y) assuming periodic boundary conditions
    # this is used in the data-augmentation step, i.e. when sampling from the maps, we have the option to
    # generate a random translation vector and apply it, this increases the effective number of possible maps
	def translate_maps(self, maps, t_x, t_y):
		maps = np.asarray(maps)
		slice_1 = np.concatenate([maps[:, t_x:, t_y:], maps[:, t_x:, :t_y]], 2)
		slice_2 = np.concatenate([maps[:, :t_x, t_y:], maps[:, :t_x, :t_y]], 2)
		return np.concatenate([slice_1, slice_2], 1)

	def divide_std(self,t_maps,xH_maps):
		t_maps /= np.std(t_maps)
		xH_maps /= np.std(xH_maps)
		return t_maps, xH_maps
	
	# this is the most important function of the DataFeeder object, it samples some number of maps from the
    # datasets, combines the string components with the gaussian components using the given Gmu and formats everything
    # in the way which pytorch expects
	# at present, verification is broken and cannot be used concurrently with random_indices
	def get_batch(self, start_index=0, batch_size=1, noise=None, random_indices=False, gpu_flag=False):
        # don't try to get more maps than we have

		if batch_size > self.split_index:
			print("batch size bigger than available data, exiting")
			return

		# we select a sub-array of size batch_size from map pool
		delta_T_start_index = start_index
		xH_start_index = start_index

		if random_indices:
			delta_T_start_index = np.random.choice(self.map_shape[0] - batch_size)
			xH_start_index = delta_T_start_index

		delta_T_maps_batch = self.train_T[delta_T_start_index:delta_T_start_index + batch_size]
		xH_maps_batch = self.train_xH[xH_start_index:xH_start_index + batch_size]

		validation_delta_T_maps_batch = self.val_T
		validation_xH_maps_batch = self.val_xH

		if random_indices:
			t_x = np.random.choice(200)
			t_y = np.random.choice(200)
			delta_T_maps_batch = self.translate_maps(delta_T_maps_batch, t_x, t_y)
			xH_maps_batch = self.translate_maps(xH_maps_batch, t_x, t_y)

        # we normalize the final maps by dividing by the standard deviation, this is because neural nets behave better
        # when we feed them inputs of order 1, feeding them a map with std=1e-5 would make it hard for the training to
        # converge
		#delta_T_maps_batch, xH_maps_batch = self.divide_std(delta_T_maps_batch, xH_maps_batch)
		
        # if a gpu is detected, it sends the batch to
        # gpu memory
		
		if gpu_flag or t.cuda.is_available():
			return delta_T_maps_batch.cuda(), xH_maps_batch.cuda(), validation_delta_T_maps_batch.cuda(), validation_xH_maps_batch.cuda()
		else:
			return delta_T_maps_batch, xH_maps_batch, validation_delta_T_maps_batch, validation_xH_maps_batch
		

if __name__ == '__main__':
	feeder = DataFeeder(prefix='sweep-20', verbose=True)
	inputs, answers, val_in, val_ans = feeder.get_batch(batch_size=1)
	print(np.shape(inputs))
	plt.imshow(inputs[0], cmap=plt.cm.hot)
	plt.show()