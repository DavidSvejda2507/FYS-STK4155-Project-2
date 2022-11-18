# FYS-STK4155-Project-2

Codebase for the second FYS-STK project. All of the actual calculations were done from the Jupyter notebook `Notebook.ipynb`, the rest of the python files are either tests or functions that are used by the notebook.

## NeuralNet.py

Class for a Neural network.

## ActivaionFunctions.py

Collection of functions for making the activation functions that the NN needs

## optimisers.py

Collection of optimisers for determining the parameters of a neural network

## lrSchedules.py

Collection of functions that define learning rate schedules for the neural network learningrate optimiser

## Data.py

Function for importing the dataset from sklearn

## Hyper.py

Functions for optimising the hyperparameters

## plot.py

Functions for plotting the results of Hyper.py

## validate.py

Function for testing a network on the validation data and plotting the results

## plotweights.py

Function for plotting the weights of the first layer of a network

## solver.py

Gradient Descent code for simple learning problems

## tests.py

Some tests to confirm that the optimisers and networks are working