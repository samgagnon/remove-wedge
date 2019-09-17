"""
by Samuel Gagnon-Hartman, 2019

This script transforms files of a specified type within an associated database
and then saves these files back into the database with a prefix added to the filename.

The transformations applied are performed in Fourier space and are meant to 
replicate the distortions and limitations of 'actual' datasets.

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


def multivariate_normal(x,y,z,u,std):
    return np.exp(-1*((x-u)**2+(y-u)**2+(z-u)**2)/(2*std**2))

def gaussian_builder(dim,std):
    u = int(dim/2)
    gaussian_array = np.zeros([dim,dim,dim])
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                gaussian_array[x,y,z] = multivariate_normal(x,y,z,u,std)
    print("GAUSSIAN OF STD", std, "GENERATED")
    return gaussian_array

def reader(filename):
    """
    Reads a single file and converts it into a numpy array.
    """
    f = open(filename, "rb")
    data = f.read()
    f.close()
    _data = np.frombuffer(data, np.float32)
    print(_data)
    print(len(_data))
    print(type(_data[0]))
    if sys.byteorder == 'big':
        _data = _data.byteswap()
    DIM = int("" + filename.split("_")[-2])
    _data.shape = (DIM,DIM,DIM)
    _data = _data.reshape((DIM,DIM,DIM), order = 'F')
    return _data

def fourier(x):
    """
    Applies a fourier transform to a 2D slice of an input map.
    """
    output = np.fft.fftshift(np.fft.fftn(x))
    return output

def sweep(x, alpha, fill=False):
    """
    Applies blind cones to Fourier space.
    -alpha: the external angle of the cone w.r.t. the z=0 plane
    -fill: fills wedge with Gaussian noise if True
    """
    DIM = np.shape(x)[0]
    mid = int(DIM/2)
    real_mean = np.mean(np.real(x))
    real_std = np.std(np.real(x))
    im_mean = np.mean(np.imag(x))
    im_std = np.std(np.imag(x))
    for i in range(DIM):
        z_distance = abs(mid-i)
        radius = np.tan(alpha*np.pi/180)*z_distance
        for j in range(DIM):
            for k in range(DIM):
                xy_distance = np.sqrt((j-mid)**2+(k-mid)**2)
                if xy_distance>radius:
                    if fill:
                        x[j,k,i] = np.random.normal(real_mean, real_std) + np.random.normal(im_mean,im_std)*1j 
                        if i%10 is 0 and j%10 is 0 and k%10 is 0:
                            print("GAUSSIAN SAMPLE:", x[j,k,i])
                    else:
                        x[j,k,i] = 0j
    plt.imshow(np.real(x)[99])
    plt.show()
    return x

def bar(x, maximum):
    """
    Applies blind bar to Fourier space.
    """
    DIM = np.shape(x)[0]
    half = int(DIM/2)
    minimum = -1*maximum
    for i in range(DIM):
        for j in range(DIM):
            if minimum<i-half<maximum:
                x[i][j][:] = 0j
    #plt.imshow(np.real(x)[:,:,99])
    #plt.show()
    return x

def gaussian(x, std):
    """
    Applies Gaussian bias to Fourier space.
    """
    x = x*prebuilt_gaussian
    #plt.imshow(np.real(x)[:,:,99])
    #plt.show()
    return x

def inverse(x):
    """
    Applies inverse Fourier transform to return to 2D spatial map.
    """
    output = np.fft.ifftn(np.fft.ifftshift(x))
    return output

def writer(x, mode, tag, filename, file_subfolder, data_folder):
    """
    Writes transformed data cube to file.
    """
    data_in_bytes = x.flatten().tobytes()
    if data_folder=='main':
        prefix = ''
    elif data_folder=='yue':
        prefix = '..' + '\\' + 'yue_data'
    modified_filename = prefix + file_subfolder + '\\' + mode + '-' + tag + '-' + filename
    print(modified_filename)
    write_to = open(modified_filename, 'wb')
    write_to.write(data_in_bytes)
    write_to.close()

def fourier_main(file_prefix, _sweep=False, _bar=False, _gaussian=False, fill=False, max_files=1,
                std=1, alpha=1, bar_max=10, data_folder='main'):
    """
    main method

    -file_prefix: prefix of files to be transformed
    -sweep: True if sweep applies
    -bar: True if bar applies
    -gaussian: True if gaussian applies
    -fill: fills wedge with gaussian noise on sweep transform, no effect for other transforms
    -max_files: max number of files to convert
    -data_folder: can either be main or yue, determines where files are saved to
    """
    print(fill)
    files_converted = 0
    assert data_folder=='main' or data_folder=='yue'

    if _gaussian:
        mode = 'gauss'
        tag = str(std)
        global prebuilt_gaussian
        prebuilt_gaussian = gaussian_builder(200, std) # fix this to make it more modular
        prebuilt_gaussian = prebuilt_gaussian.astype(np.float32)
    if _bar:
        mode = 'bar'
        tag = str(bar_max)
    if _sweep:
        if fill:
            mode = 'fill'
        else:
            mode = 'sweep'
        tag = str(alpha)

    for directory in os.listdir():
        if basename(directory)[0] == 'z' and basename(directory)[2] != '1' and basename(directory)[2] != '7':
        #if basename(directory)[0] == 'z':
            file_folder = '.' + '\\' + directory
            for folder in os.listdir(file_folder):
                if basename(folder)[0:3] == 'Run':
                    file_subfolder = file_folder + '\\' + folder
                    for filename in os.listdir(file_subfolder):
                        # this is where we load the maps
                        prefix = "" + filename.split("_")[0]
                        second = "" + filename.split("_")[1]

                        # conditional naming for gaussianized files
                        if prefix=='gaussianized-delta' and mode=='sweep':
                            result_name = 'swg' + '-' + tag + '-' + filename
                        else:
                            result_name = mode + '-' + tag + '-' + filename
                        
                        already_exists = False
                        if data_folder=='main':
                            for other_files in os.listdir(file_subfolder):
                                if other_files==result_name:
                                    already_exists = True
                                    break
                        if data_folder=='yue':
                            other_file_subfolder = '..' + '\\' + 'yue_data' + file_subfolder
                            for other_files in os.listdir(other_file_subfolder):
                                if other_files==result_name:
                                    already_exists = True
                                    break


                        # if file_prefix==prefix and files_converted<max_files and not already_exists: # use commented parts for delta_T
                        if file_prefix==prefix and files_converted<max_files and second=="T" and not already_exists:
                            files_converted+=1
                            print("converting:", filename)
                            # slice cube and handle each slice one at a time
                            data_cube = reader(file_subfolder + '\\' + filename)
                            plt.imshow(data_cube[99], cmap=plt.cm.hot)
                            plt.show()
                            transformed_data_cube = np.zeros(np.shape(data_cube))
                            # apply fourier transform to layer
                            fourier_data = fourier(data_cube)
                            plt.imshow(np.real(fourier_data[99]))
                            plt.show()
                            # apply chosen transformations to layer
                            if _gaussian:
                                fourier_data = gaussian(fourier_data, std)
                            if _bar:
                                fourier_data = bar(fourier_data, bar_max)
                            if _sweep:
                                fourier_data = sweep(fourier_data, alpha, fill=fill)
                            # apply inverse fourier transform to layer
                            transformed_data_cube = np.real(inverse(fourier_data))
                            # convert to proper data type
                            transformed_data_cube = transformed_data_cube.astype(np.float32)
                            plt.imshow(transformed_data_cube[99], cmap=plt.cm.hot)
                            plt.show()
                            writer(transformed_data_cube, mode, tag, filename, file_subfolder, data_folder)
                            print(filename, "written to file!")
                            print("files converted:", files_converted)

if __name__ == "__main__":
    #alpha = np.arctan(0.12/0.4)*180/np.pi
    fourier_main('delta', _sweep=True, _bar=False, _gaussian=False, fill=True, max_files=3,
                std=80, alpha=10, bar_max=80, data_folder='main')