"""
by Samuel Gagnon-Hartman, 2019

Deletes unnecessary files from database. Looks for files generated with a
specific tag.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import os
import io
import sys

from os.path import basename
from torch.autograd import Variable
from scipy.misc import imshow
from scipy import signal

def cleaner(bar=False, gaussian=False, sweep=False, tag=0, data_folder='main'):
    """
    main method

    bar, gaussian, sweep: three tags which may be searched for deletion
    
    Only one tag may be used.
    """
    tag = str(tag)
    assert data_folder=='main' or data_folder=='yue'
    if gaussian:
        mode = 'gauss'
        assert not (bar or sweep)
    if bar:
        mode = 'bar'
        assert not (sweep or gaussian)
    if sweep:
        mode = 'sweep'
        assert not (bar or gaussian)

    if data_folder=='main':
        look_here = '.'
    elif data_folder=='yue':
        look_here = '..' + '\\' + 'yue_data'

    for directory in os.listdir(look_here):
        if basename(directory)[0] == 'z' and basename(directory)[2] != '1' and basename(directory)[2] != '7':
            file_folder = look_here + '\\' + directory
            for folder in os.listdir(file_folder):
                if basename(folder)[0:3] == 'Run':
                    file_subfolder = file_folder + '\\' + folder
                    for filename in os.listdir(file_subfolder):
                        # this is where we load the maps
                        prefix = "" + filename.split("_")[0]
                        second = "" + filename.split("_")[1]
                        split_prefix = prefix.split("-")
                        file_mode = ""
                        file_tag = ""
                        file_type = ""
                        if len(split_prefix)>2:
                            #print(split_prefix)
                            file_mode = "" + split_prefix[0]
                            file_tag = "" + split_prefix[1]
                            file_type = "" + split_prefix[2]
                        # file_tag = "" + prefix.split("-")[1]
                        # if file_prefix==prefix and files_converted<max_files and not already_exists:
                        # use commented parts for delta_T
                        if file_type=="delta" and file_mode==mode and file_tag==tag and second=="T":
                            print(filename)
                            file_location = file_subfolder + '\\' + filename
                            os.remove(file_location)

if __name__=='__main__':
    cleaner(sweep=True, tag=70, data_folder='yue')                                                                                                                               