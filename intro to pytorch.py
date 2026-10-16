import torch
import math

# a one dimensional tensor is a vector, two dimensional is a matrix
# x = torch.empty(3,4)
# print(x)

# zeros = torch.zeros(2, 3)
# print(zeros)

# ones = torch.ones(2, 3)
# print(ones)

x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.ones_like(x)
print(x.shape)
print(x)