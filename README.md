
# LR model 

This "LR model" demonstrates how logistic regression works. I wrote this for challenging myself to deal with all the details from scratch.

Logistic regression is a classification algorithm. LR assume data follow Bernoulli Distrbution and use sigmoid as it's link funcion (see the section below - why sigmoid function - to get more information).

## What's New

```
This code separate model to several part, including section of `data preparing`, section of `backpropagation and optimization` and section of `activation`. 
```
* The `load_data` functrion is for preprocessing the training/testing corpuse. It's do not require any arguemnts (usefixed path). Return Y for test, X for test, classes list, Y for training and X for training, respectively.
* The `activator` functrion is for the activateon purpose. It requires 3 arguments including weights, X and bisa, respectively. Return the resulf from sigmoid function.
* The `sigmoid` functrion is an assistant for 'activator'. It requires 1 argument - z, which is dot(W.T, X) +b. Return the result of z. 
* The `optimize`functrion is for training and optimize the cost using technigue of gradient descent. It requires 6 argumens including weights, bias , X, Y, expected learning rate and expected iteration number, respectively. Return a updated weights, a updated bias and its cost.
* The `predict` functrion is for predicting, use `0.5` as its threshold. It requires 3 arguments trained weights, trained bias and input X, respectively. Return the final result of classification. 
* The `propagate` functrion is an assistant function for 'optimize'.  It requires wights, bias, X and Y, respectively. Return db, dw and cost. 
* The `run` functrion is the key function and collects above functions together. In this function you can see the training/testing process step by step. It requires 6 arguments X for training, Y for training, X for testing, Y for testing, expected iteration number and expected learning rate, respectively. 


### Why sigmoid ?

* Because sigmoid is the `canonical linked function of Bernoulli's principle`. One distribuition can have many linked function as long as it follow: 1. Can be inversed 2.Domian is from {-inf, inf}/ 3. Range is (0,1). However, cononical linked function just have one. Acturally, the inversted function of canonical linked function is `activate function`. 
* canonical linked function can be more efficient because it's sufficient statistic on average.


## Requirements

LR model has been tested on Python 2.7.

### Prparing Dependencies

* miniconda (https://docs.conda.io/en/latest/miniconda.html)

## Building LR model

### Install all required dependencies

```
conda env create -f freeze.yml
```
