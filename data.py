import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from test_file import NeuralNetwork
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Let's load our data
data = pd.read_csv("Projet Alami/iris.csv", encoding='utf-8')
cols = data.columns.to_list()

# Let's encode our classes to numeric values 
classes = data['class'].unique().tolist()
data['class'] = data['class'].apply(lambda cls : classes.index(cls))


'''# now we one hot encode our classes to binary codes
encoded_labels = pd.get_dummies(data['class'], prefix= 'flower', dtype=int)
data[encoded_labels.columns] = encoded_labels
new_cols = cols[:-1]+encoded_labels.columns.tolist()
final_data = data[new_cols]'''



data = data.to_numpy()
X, y = data[:, :-1], data[:, -1]

#NN = NeuralNetwork(X, y, {0: (10, 'Relu') ,1: (10, 'Relu'), 2:(3, 'Softmax')}, batch_size=40, num_flies=1000, DFO_bounds=2,epochs=2,delta = 0.4,max_iter=200, ratio=0.7)
#NN = NeuralNetwork(X, y, {0: (5, 'Relu') ,1: (5, 'Relu'), 2:(3, 'Softmax')}, batch_size=40, num_flies=1000, DFO_bounds=2,epochs=2,delta = 0.5,max_iter=200, ratio=0.7) ==> 0.07



NN = NeuralNetwork(X, y, {0: (20, 'Relu'), 1: (20, 'Relu') , 2:(3, 'Softmax')}, batch_size=40, num_flies=5000, DFO_bounds=1,epochs=1,delta = 0.5,max_iter=20, ratio=0.6)
hope = NN.train_All_data_at_once()
