mport pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
	
'''
ComputeCost computes the cost function
'''
def computeCost(X, y, theta):
	#computeCost Compute cost for linear regression
	#   J = computeCost(X, y, theta) computes the cost of using theta as the
	#   parameter for linear regression to fit the data points in X and y

	# Initialize some useful values
	m = len(y); # number of training examples

	# You need to return the following variables correctly 
	J = 0;

	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost of a particular choice of theta
	# =========================================================================
	# You should set J to the cost.
	
	
	return J

'''
gradientDescent function iterates till it finds a minima
'''
def gradientDescent(X, y, theta, alpha, num_iters):
	#function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
	#GRADIENTDESCENT Performs gradient descent to learn theta
	#   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
	#   taking num_iters gradient steps with learning rate alpha

	# Initialize some useful values
	m = len(y); # number of training examples
	J_history = np.zeros((num_iters, 1));

	for iter in range(num_iters):

		# ====================== YOUR CODE HERE ======================
		# Instructions: Perform a single gradient step on the parameter vector
		#               theta. 
		#
		# Hint: While debugging, it can be useful to print out the values
		#       of the cost function (computeCost) and gradient here.
		#
		
		# Plot the linear fit
		if(iter%200==0):
			plt.scatter(X_data, y_data, marker='o',  color='g', label='orig') 
			y_data_predicted = np.matmul(X,theta)
			plt.plot(X_data, y_data_predicted, marker='*', linestyle='-', color='b', label='pred')
			plt.legend(loc='lower right')
			plt.show(block=False)
			time.sleep(3)
			plt.close()




		# ============================================================

		# Save the cost J in every iteration    
		J_history[iter] = computeCost(X, y, theta)
		print "Cost @ iteration: ",iter, " = ", J_history[iter]
	return theta

#Load Data
data 	= pd.read_csv('ex1data1.txt', header =  None, names = ['Population', 'Profits'])
y_data 	= data.iloc[:,1]
X_data 	= data.iloc[:,0]

m 		= len(y_data)						  #Number of training samples
y 		= np.array(y_data).reshape(m,1)		
X 		= np.c_[np.ones(m), np.array(X_data)] # Add a column of ones to x

# Plot the Initial Data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Population - Profit Scatter Plot')
ax.set_xlabel('Population in 10000s')
ax.set_ylabel('Profit in 10000$')
plt.scatter(X_data, y_data, marker='o',  color='g', label='orig') 
plt.show()
#quit()

theta 	= np.zeros((2, 1)).reshape((2,1)) # initialize fitting parameters
print "Cost Function Value is:", computeCost(X, y, theta)
#quit()

# Some gradient descent settings
#theta 	= np.array([40,40]).reshape((2,1))# Try initializing from a different point. The convergence will be seen easily
iterations = 1500;
alpha = 0.01;

# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

# print theta to screen
print 'Theta found by gradient descent: ', theta.item(0), theta.item(1)

# Plot the linear fit
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Population - Profit Scatter Plot')
ax.set_xlabel('Population in 10000s')
ax.set_ylabel('Profit in 10000$')
plt.scatter(X_data, y_data, marker='o',  color='g', label='orig') 
y_data_predicted = np.matmul(X,theta)
plt.plot(X_data, y_data_predicted, marker='*', linestyle='-', color='b', label='pred')
plt.legend(loc='lower right')




