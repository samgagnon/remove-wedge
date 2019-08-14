# 21cm Wedge Remover

A Convolutional Neural Network for recovering 21cm intensity information lost to "The Wedge".

## Dependencies
* **python** >= 3.5
* **numpy**
* **scipy**
* **matplotlib**
* **pytorch**
* **cuda** for GPU. The program will automatically use it if available.

## Files in src
* **config.py** - Contains the settings on variables relevant to the rest of the code. This should be the only file you need to edit to run the code.

* **model_def.py** - The definition of the network model.

* **train.py** - Use this to train the network from scratch or to continue training. Defaults to GPU if available.

* **feeder.py** - Defines a DataFeeder class which loads desired input and target maps, then provides a method for sampling from them. This is mainly used by train.py to get the training data.

* **utils.py** - Miscellaneous utilities used in the code. 

* **statistics.py** - Contains statistical methods used in qualifying the performance of the network.

* **verification.py** - Used in train.py to compute validation loss, and may be called on its own to calculate the validation loss on a pre-trained network.

## Files in data
* **fourier.py** - Applies wedge effects to data stored in the folder.

* **cleaner.py** - Removes files of a specified extension from database.

## Folders in data
Data files are organized within the directories below
```
data/redshift/Run x - RNG y/files
```
## Training the Network
After navigating to /src in terminal, the following line will initiate the training of a new network.

```
python train.py sweep-10
```

The first command after the filename specifies the desired input maps. The target map is set in config.py.

## Generating New Transformed Files
**fourier.py** offers three types of transformations which may be applied to existing 21cm boxes. These are **sweep**, **gaussian**, and **bar**. **sweep** is representative of "The Wedge", **gaussian** multiplies the map's Fourier profile by a Gaussian distribution, and **bar** removes a bar of a specified width in Fourier space.

To generate a transformed file from an existing data file, edit the main method of **fourier.py** to perform the transformation you desire. To perform a certain transformation, then set that argument to **True** in the main method. Each method has a corresponding variable which needs to be set before running, so set that as well. Then, navigate to /data in terminal and run the following line.
```
python fourier.py
```
