import numpy as np
import tflearn
from tflearn.datasets import titanic
from tflearn.data_utils import load_csv


titanic.download_dataset('titanic_dataset.csv')

data, labels = load_csv('titanic_dataset.csv', target_column = 0,
						categorical_labels = True, n_classes = 2)

#mapping fields to numerical values. Delete unneccary fields

def process_numericals(passengers, delete_columns):
	for delete in sorted(delete_columns, reverse = True):
		[passenger.pop(delete) for passenger in passengers]
	for i in range(len(passengers)):
		passengers[i][1] = 1. if passengers[i][1] == 1 else 0

	return np.array(passengers, dtype = np.float32)

ignore_column = [1,6]

data = process_numericals(data, ignore_column)
print(data[1])
#neural net layers

net = tflearn.input_data(shape = [None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation = 'softmax')
net = tflearn.regression(net)

#DNN does training, prediction and save

model = tflearn.DNN(net)

model.fit(data, labels, n_epoch = 10, batch_size = 32, show_metric= True)
