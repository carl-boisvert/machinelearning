import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#Import dataset with Panda
dataset = pd.read_csv("Data.csv")

#Independant variables
x = dataset.iloc[:, :-1].values
#Dependant variables
y = dataset.iloc[:, 3].values

#Create test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

print("Done")