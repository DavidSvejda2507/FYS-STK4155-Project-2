# FYS-STK4155-Project-2

Codebase for the second FYS-STK4155 project. We train a neural network to recognize the hand-written digits from the MNIST dataset.
All of the actual calculations were done from the Jupyter notebook `Notebook.ipynb`, therefore you can simply run the notebook cell by cell to reproduce our results. The rest of the python files are either tests or functions that are used by the notebook.

### Requirements

In order to run the codes, you need python and in particular the `scikitlearn` library, which is used to import the MNIST dataset.


### Contents of the repository

*  ` NeuralNet.py `: class for a Neural network.

*  ` ActivaionFunctions.py `: collection of functions for making the activation functions that the NN needs.

*  ` optimisers.py `: collection of optimisers for determining the parameters of a neural network.

*  ` lrSchedules.py `: collection of functions that define learning rate schedules for the neural network learningrate optimiser.

*  ` Data.py `: function for importing the MNIST dataset from sklearn

*  ` Hyper.py `: functions for optimising the hyperparameters

*  ` plot.py `: functions for plotting the results of Hyper.py `

*  ` validate.py `

*  ` /Data `: a folder containing both the plots and the data which are used to do the plots. The data are organized into subfolders,
and they are saved in `numpy` format. 

Function for testing a network on the validation data and plotting the results

*  ` plotweights.py `

Function for plotting the weights of the first layer of a network

*  ` LinearGDtest.py `, `LinearModel.py `, `toy-gradient-descent.ipynb`

Functions for testing the optimisers on a simple test case and notebook for generating the plots

### Authors

<p align="center">
<button  style="border: transparent; background-color: transparent;">
    <img align="left" width=10% src="https://avatars.githubusercontent.com/u/89582342?v=4"> 
</button> [Gianmarco](https://github.com/giammy00) <br>
<button style="border: transparent; background-color: transparent;">
    <img align="left" width=10% src="https://avatars.githubusercontent.com/u/111510879?v=4"> 
</button>
<button style="border: transparent; background-color: transparent;">
    <img align="left" width=10% src="https://avatars.githubusercontent.com/u/74814411?v=4"> 
</button>
</p> 










