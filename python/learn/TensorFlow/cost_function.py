
import numpy
import matplotlib.pyplot as plt
import GradientDescent as gd

# Parameters
learning_rate = 0.5
n = 12
theta= -2
thetaValues =[]
costValues=[]

# Training Data
train_X = numpy.asarray([1, 2, 3, 4, 5 , 6])
train_Y = numpy.asarray([1, 2, 3, 4, 5 , 6])

for index in range(n):
    predictedY = train_X * theta;
    cost = gd.compute_error_for_given_points(theta,0,train_X,train_Y)
    thetaValues.append(theta)
    costValues.append(cost)
    theta = theta + learning_rate;

minCost = numpy.min(costValues) 
print("minimum Cost = %d",minCost)

minCostIndex = costValues.index(minCost)
thetaFinal = thetaValues[minCostIndex]
print("Final Theta value = %d",thetaFinal)

#gradient descent
m,b,mvalues,bvalues,cValues = gd.gradient_descent_runner(train_X,train_Y, -2, 0, 0.005, 1000)

print(m)
print(b)

# display the cost curve
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(thetaValues,costValues,c="Red")
plt.scatter(thetaValues,costValues,c="Red")
plt.scatter(mvalues,cValues,c="Yellow")
plt.title("cost graph")
plt.xlabel("theta")
plt.ylabel("cost")



# display the Linear Regression Result
plt.subplot(1,2,2)
plt.scatter(train_X,train_Y,c="blue")
plt.plot(train_X, train_X*thetaFinal,c="yellow")
plt.title("Linear Regression Result")
plt.show()