# Logistic-Regression 

This "Logistic-Regression" demonstrates how logistic regression works. I wrote this for challenging myself to deal with all the details from scratch.
Logistic regression is a classification algorithm. LR assume data follow Bernoulli Distribution and use sigmoid as its link funcion (see the section below - why sigmoid function - to get more information).


## What's New
```Use several functions to demo the different stage of model training. ```

* The `load_data` function is for preprocessing the training/testing corpus. It does not require any arguments (usefixed path). Return Y for test, X for test, classes list, Y for training and X for training, respectively.

* The `activator` function is for the activation purpose. It requires 3 arguments including weights, X and bisa, respectively. Return the result from sigmoid function.

* The `sigmoid` function is an assistant for 'activator'. It requires 1 argument - z, which is dot(W.T, X) +b. Return the result of z. 

* The `optimize`functrion is for training and optimizing the cost using the technique of gradient descent. It requires 6 arguments including weights, bias , X, Y, expected learning rate and expected iteration number, respectively. Return updated weights, an updated bias and its cost.

* The `predict` function is for predicting, using `0.5` as its threshold. It requires 3 arguments: trained weights, trained bias and input X, respectively. Return the final result of classification. 

* The `propagate` function is an assistant function for 'optimize'.  It requires wights, bias, X and Y, respectively. Return db, dw and cost. 

* The `run` function is the key function and collects above functions together. In this function you can see the training/testing process step by step. It requires 6 arguments X for training, Y for training, X for testing, Y for testing, expected iteration number and expected learning rate, respectively. 

### Why sigmoid ?

* Because sigmoid is the `canonical linked function of Bernoulli's principle`. One distribution can have many linked functions as long as it follows: 1. Can be inverted 2.Domain is from {-inf, inf}/ 3. Range is (0,1). However, canonical linked functions just have one. Actually, the inverse function of the canonical linked function is the `activate function`. 

* canonical linked function can be more efficient because it's a sufficient statistic on average.

## Requirements
LR model has been tested on Python 2.7.

### Preparing Dependencies
* miniconda (https://docs.conda.io/en/latest/miniconda.html)

### Install all required dependencies
```conda env create -f freeze.yml```
