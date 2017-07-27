import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Import dataset with Panda
dataset = pd.read_csv("Salary_Data.csv")

#Independant variables
x = dataset.iloc[:, 0].values
#Dependant variables
y = dataset.iloc[:, 1].values

#Create test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

#Fit the test set to the the training set

x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)

#Predict
y_pred = linear_regressor.predict(x_test.reshape(-1,1))

#Visualize data
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, linear_regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

#Visualize Test data
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, linear_regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
print("Done")