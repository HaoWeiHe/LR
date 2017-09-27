import numpy as np
import h5py 
import math


def load_data():

	#load for test dataset
	f = h5py.File("../datasets/test_catvnoncat.h5", "r")
	test_set_x_orig = f["test_set_x"]
	test_set_y_orig = f["test_set_y"]
	list_classes = f["list_classes"]
	test_set_x = np.array(test_set_x_orig)
	test_set_y = np.array([test_set_y_orig[:]])

	#load for training dataset
	f = h5py.File("../datasets/train_catvnoncat.h5", "r")
	train_set_x_orig = f["train_set_x"]
	train_set_y_orig = f["train_set_y"]
	train_set_y = np.array([train_set_y_orig[:]])
	train_set_x = np.array(train_set_x_orig)

	return test_set_y, test_set_x, list_classes, train_set_y, train_set_x


def sigmoid(z):
	return  1.0 /(1 + np.exp(-1*z))


def activator(W, X, b):
	
	return sigmoid(np.dot(W.T, X) + b)

def optimize(W,b, X,Y, learning_rate,  iteration_number):
	
	lost = 0 
	num_m = X.shape[0]
	
	# w := w - learning_rate.dw
	
	for i in range(0, iteration_number):
		paramter , cost= propagate(W,b,X,Y)
		db = paramter['db']
		dw = paramter['dw']
		
		W = W - np.dot(learning_rate, dw)
		b = b - np.dot(learning_rate, db)
		
		
		if i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))
	
	# lost = cost

	return W, b, lost


def predict(W, b, X):
	num_m = X.shape[1]
	Y_prediction = np.zeros((1, num_m))

	# remember that activator(W,X, b ) return a matric
	A = activator(W,X,b)

	for i in range(0, num_m):
		Y_prediction[0][i] = 1 if A[0][i] > 0.5 else 0
	
		
	return Y_prediction

def propagate(W,b,X,Y):
	
	num_m = X.shape[1]

	
	A = activator(W,X,b)
	
	
	cost = -1.0/num_m * (np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T))  
	
	### squeeze : 1*m -> m (change to scale) / m*1 -> m (change to scale) / 3*m -> 3*m(no change)
	cost = np.squeeze(cost)
	

	dw = np.dot(X, (A-Y).T) / num_m
	db = np.sum(A-Y)/ num_m
 	

	paramter = {'db':db, 'dw':dw}


	return paramter, cost

def run(train_set_x, train_set_y, test_set_x, test_set_y, iteration_number , learning_rate ):

	# reshape X to dim (wid * len * 3, m ) 
	test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0] , -1).T
	train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0] , -1).T
	
	test_set_x_flatten = test_set_x_flatten / 255.
	train_set_x_flatten = train_set_x_flatten / 255.

	# dim(X) = (n, m ) -> dim(w) = (n, 1)
	W, b = np.zeros((train_set_x_flatten.shape[0],1)), 0
	
 
	# gradient decent
	W, b, lost = optimize(W,b, train_set_x_flatten, train_set_y, learning_rate,  iteration_number)


	Y_prediction_train = predict(W, b, train_set_x_flatten)
	Y_prediction_test = predict(W, b, test_set_x_flatten)
	

	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))
	

if __name__ == '__main__':

	test_set_y, test_set_x, list_classes, train_set_y, train_set_x = load_data()
	run(train_set_x, train_set_y, test_set_x, test_set_y, 2000, 0.005)