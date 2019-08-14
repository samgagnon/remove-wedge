# 21cm Wedge Recovery

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
```
data/redshift/Run x - RNG y/files
```
## Training the Network
```
python train.py sweep-10
```

The first command after the filename specifies the desired input maps. The target map is set in config.py.
