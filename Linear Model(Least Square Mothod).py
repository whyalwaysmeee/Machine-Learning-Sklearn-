import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
price = datasets.load_boston()

# Use only one feature
price_x = price.data[:,np.newaxis,5]

# Split the data into training/testing sets
price_x_train = price_x[:-20]
price_x_test = price_x[-20:]
# Split the targets into training/testing sets
price_y_train = price.target[:-20]
price_y_test = price.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(price_x_train, price_y_train)
# Make predictions using the testing set
price_y_pred = regr.predict(price_x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(price_y_test, price_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(price_y_test, price_y_pred))

# Plot outputs
plt.scatter(price_x_test, price_y_test,  color='black')
plt.plot(price_x_test, price_y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
#display the plot graph
plt.show()




