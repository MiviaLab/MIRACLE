import numpy as np
import os

path_first = 'Test_Sets/test_OpenBMI_first.npz'
dataset_first = np.load(path_first, allow_pickle=True)
data_first = dataset_first['data']
labels_first = dataset_first['labels']

path_last = 'Test_Sets/test_OpenBMI_last.npz'
dataset_last = np.load(path_last, allow_pickle=True)
data_last = dataset_last['data']
labels_last = dataset_last['labels']

data = np.concatenate((data_first, data_last))
labels = np.concatenate((labels_first, labels_last))

np.savez('Test_Sets/test_OpenBMI.npz', data=data, labels=labels)

os.remove(path_first)
os.remove(path_last)
