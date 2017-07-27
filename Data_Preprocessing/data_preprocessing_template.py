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

#Deal with empty value by making them the mean, which give them a value but won't influence the data result
imputer = Imputer()
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])

#Deal with categorical data. "Spain" or "France" as no meaning as they are not numbers. We need to change them to numbers
label_encoder = LabelEncoder()
x[:,0] = label_encoder.fit_transform(x[:,0])
#Now we need to make sure there's no hierachie between the categorical values
one_hot_encoder = OneHotEncoder(categorical_features=[0])
x = one_hot_encoder.fit_transform(x).toarray()

#Deal with categorical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Create test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#Feature scaling
standard_scaler_x = StandardScaler()
x_train = standard_scaler_x.fit_transform(x_train)
x_test = standard_scaler_x.transform(x_test)

print("Done")