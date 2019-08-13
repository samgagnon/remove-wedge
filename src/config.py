"""
by Samuel Gagnon-Hartman, 2019

This file contains global variables important in other files. To import, use 
from config import *.
"""
import torch as t

map_number = 1

#DataFeeder
#The location of the directory containing training files
backslash = '\\'
training_data_folder = '..' + '\\' + 'yue_data' + '\\'
xH_maps_max = map_number*4 if map_number*4<60 else 60
delta_T_maps_max = map_number*4 if map_number*4<60 else 60

# train.py
# The default network type
default_network_type = 'Net'
models_directory = '..' + '\\' + 'models'
images_directory = '..' + '\\' + 'images'
start_training_from_this_model_filename = None
learning_rates_list = [1e-3,1e-4,1e-5,1e-6]
num_steps_per_train_run_list = [5000] + [2000]*(len(learning_rates_list)-1) if t.cuda.is_available() else [500] + [1000]*len(learning_rates_list)
save_model_and_images_every = 10 if t.cuda.is_available() else 20
batch_size = 720 if t.cuda.is_available() else map_number*4
border = 25
log_file_location = '..' + '\\' + 'training_log_file'
dropout_rate = 0.2
strikes_max = 5
load_params_max = 3

# verification.py
save_verification_to = '..' + '\\' + 'verification'
model_filename = '..' + '\\' + 'models' + '\\' + 'great_sweep_model.pth'