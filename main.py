import torch
import numpy as np

x = torch.rand(24, 24, 2, 3)
print(x.shape)
x = torch.transpose(x, -3, -1)
print(x.shape)

data = np.loadtxt('data/training_arrays.txt', dtype='i')
print(data[0])
print(len(data))