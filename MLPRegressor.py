import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

file = pd.read_csv(r'timesereis_8_2.csv')
#print(file)

inputs = file[['0', '1', '2', '3','4','5','6','7']]
targets = file[['8','9']]

# print(inputs.shape)
# print(targets.shape)
x_train ,x_test , y_train , y_test = train_test_split(inputs,targets,test_size=0.2,random_state=101)

mlpReg = MLPRegressor(hidden_layer_sizes=(500,50),learning_rate_init=0.001,early_stopping=True,verbose=True)
mlpReg.fit(x_train,y_train)
predictions = mlpReg.predict(x_test)

print(mlpReg.score(x_test,y_test))
print(metrics.mean_absolute_error(y_test, predictions))

#print(metrics.mean_absolute_error(y_test,predictions))
#print(metrics.r2_score(y_test,predictions))