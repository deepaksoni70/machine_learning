import numpy as np

def gradient_descent_runner(xValues, yValues, starting_m, starting_b, learning_rate, num_iterations):
    m = starting_m
    b = starting_b

    mValues=[]
    bValues=[]
    costValues=[]

    for i in range(num_iterations):
        m, b = step_gradient(m, b, xValues, yValues, learning_rate)
        mValues.append(m)
        bValues.append(b)
        costValues.append(compute_error_for_given_points(m,b,xValues,yValues))

    return m, b, mValues, bValues, costValues

def step_gradient(current_m, current_b, xValues, yValues, learning_rate):
    #gradient descent
    m_gradient = 0
    b_gradient = 0

    #Calculate optimal values for model

    #To calculate the gradient we need to calculate 
    #the partial derivative of m and b
    n = float(len(xValues))

    sum_m = 0
    sum_b = 0

    for i in range(len(xValues)):
        sum_m += -1 * xValues[i] * (yValues[i] - (current_m * xValues[i] + current_b))
        sum_b += -1 * (yValues[i] - (current_m * xValues[i] + current_b))

    m_gradient = (2 / n) * sum_m
    b_gradient = (2 / n) * sum_b

    m_new = current_m - (learning_rate * m_gradient)
    b_new = current_b - (learning_rate * b_gradient)
    
    return m_new, b_new


def compute_error_for_given_points(m, b, xValues, yValues):
	#sum of squared errors
	sum_error = 0
	for i in range(len(xValues)):
		sum_error += (yValues[i] - (m * xValues[i] + b)) ** 2

	return sum_error / float(len(xValues))