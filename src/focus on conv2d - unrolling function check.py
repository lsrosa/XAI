#!/usr/bin/env python
# coding: utf-8

# # Check on conv2d unrolling function

# ### TODO
# 1. take input_shapes between conv layers from activation.keys()

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
os.environ['SCIPY_USE_PROPACK'] = "True"

threads = "64"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads


# In[1]:


get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt

# from libraries import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchvision.models import vgg16, vgg16_bn
from torchvision.models import VGG16_Weights
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

import pickle
import time
import numpy as np
from tqdm import tqdm
import scipy
from scipy.sparse import csr_matrix, coo_matrix

from itertools import product


# In[2]:


# original function (with AM)
def conv2d_to_mat(input_shape, weight_tensor, bias, stride=(1, 1), padding=(1, 1)):
    """
    Convert a 2D convolutional layer to a matrix form.

    Args:
    - input_shape (tuple): Shape of the input tensor (Cin, Hin, Win), where:
        * Cin (int): Number of input channels.
        * Hin (int): Height of the input feature map.
        * Win (int): Width of the input feature map.
    - weight_tensor (numpy.ndarray): The weight tensor of the convolutional layer,
      with shape (Cout, Cin, Kh, Kw), where:
        * Cout (int): Number of output channels.
        * Cin (int): Number of input channels.
        * Kh (int): Height of the kernel.
        * Kw (int): Width of the kernel.
    - bias (numpy.ndarray): The bias term for each output channel, with shape (Cout,).
    - stride (tuple): Stride of the convolution operation in the (height, width) directions.
        Default is (1, 1).
    - padding (tuple): Padding applied to the input tensor in the (height, width) directions.
        Default is (1, 1).

    Returns:
    - numpy.ndarray: Matrix representation of the convolutional layer, where each row corresponds
      to a flattened patch of the input tensor, and the last column represents the bias term.

    Note:
    - This function assumes square padding and stride.
    - Padding and stride should be specified in the order (height, width).
    """
    
    Cin, Hin, Win = input_shape
    Cout = weight_tensor.shape[0]
    kernel_shape = weight_tensor.shape[2:]
    kernel = weight_tensor
    
    Hout = int(np.floor((Hin + 2*padding[-2] - (kernel_shape[-2] - 1) -1)/stride[-2] + 1))
    Wout = int(np.floor((Win + 2*padding[-1] - (kernel_shape[-1] - 1) -1)/stride[-1] + 1))
    
    mat = np.zeros((Cout*Hout*Wout, Cin*Hin*Win + 1), dtype=int)
    
    for f, i, j, c in product(range(Cout), range(Hout), range(Wout), range(Cin)):
        row = np.zeros((Hin, Win), dtype=int)
        row[:kernel[f, c].shape[0], :kernel[f, c].shape[1]] = kernel[f, c]
        shift = i*stride[1]*Win + stride[0]*j
        irow = f*Hout*Wout + i*Hout + j
        mat[irow, c*Hin*Win:(c+1)*Hin*Win] = np.roll(row.flatten(), shift)
        mat[irow, -1] = bias[f] 
    
    return mat


# In[24]:


# modified function (to handle large matrices)
def conv2d_to_sparse(input_shape, weight_tensor, bias, stride=(1, 1), padding=(1, 1)):
    ''' 
    Convert a 2D convolution operation represented by dense weight and bias tensors
    into a sparse matrix representation using Compressed Sparse Row (CSR) format.

    Args:
    - input_shape (tuple): The shape of the input tensor in the format (Cin, Hin, Win),
      where Cin is the number of input channels, Hin is the height of the input,
      and Win is the width of the input.
    - weight_tensor (numpy.ndarray): The weight tensor of the convolution layer
      with shape (Cout, Cin, Kh, Kw), where Cout is the number of output channels,
      Kh is the kernel height, and Kw is the kernel width.
    - bias (numpy.ndarray): The bias tensor with shape (Cout,) representing the biases
      for each output channel.
    - stride (tuple): The stride of the convolution operation in the format (stride_h, stride_w).
    - padding (tuple): The padding applied to the input in the format (pad_h, pad_w).

    Returns:
    - csr_mat (scipy.sparse.csr_matrix): The sparse matrix representation of the convolution
      operation in Compressed Sparse Row (CSR) format.
    '''
    
    
    Cin, Hin, Win = input_shape
    Cout = weight_tensor.shape[0]
    kernel_shape = weight_tensor.shape[2:]
    kernel = weight_tensor
    
    Hout = int(np.floor((Hin + 2*padding[-2] - (kernel_shape[0] - 1) -1)/stride[-2] + 1))
    Wout = int(np.floor((Win + 2*padding[-1] - (kernel_shape[1] - 1) -1)/stride[-1] + 1))
    print(Hout, Wout)
    
    rows = []
    cols = []
    data = []
    
    for f, i, j, c in product(range(Cout), range(Hout), range(Wout), range(Cin)):
        row_start = f * Hout * Wout + i * Wout + j 
        col_start = c * Hin * Win 
        weight = kernel[f, c]
        
        for m in range(kernel_shape[0]):
            for n in range(kernel_shape[1]):
                row = row_start

                col = col_start + ((m * Win + n + i * stride[1] * Win + j * stride[0]) % (Win * Hin))
                # col = col_start + ((m * Win + n + i * stride[1] * Win + j * stride[0]) // stride[1])
                # col = col_start + ((m * Win + n) // stride[1] + i * Wout + j)
                # col = col_start + ((m * Win + n) // stride[1] + i * (Win // stride[1]) + j)
                # col = col_start + ((m * Win + n) // stride[1] + i * ((Win - kernel_shape[1]) // stride[1]) + j)
                # col = col_start + ((m * Win + n + i * stride[1] * (Win + 2 * padding[-1]) + j * stride[0]) % (Win * Hin))

                rows.append(row)
                cols.append(col)
                data.append(weight[m, n])

    
    # add bias as the last column
    for f in range(Cout):
        for i in range(Hout):
            for j in range(Wout):
                row = f * Hout * Wout + i * Wout + j
                rows.append(row)
                cols.append(Cin * Hin * Win)
                data.append(bias[f])

    # create COO matrix directly
    coo = coo_matrix((data, (rows, cols)), shape=(Cout * Hout * Wout, Cin * Hin * Win + 1))
    
    # convert to csr
    csr_mat = csr_matrix(coo)
    
    return csr_mat


# Example input data

# In[25]:


input_shape = (1, 3, 3)  # Cin, Hin, Win
weight_tensor = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
bias = np.array([0])
stride = (1, 1)
padding = (1, 1)


# In[26]:


input_shape = (3, 4, 4)
# weight_tensor = np.array([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]], [[[2, 2, 2], [2, 2, 2], [2, 2, 2]]]])
weight_tensor = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                          [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                          [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
bias = np.array([1, 2, 3])  


# In[27]:


weight_tensor.shape


# In[28]:


weight_tensor[0]


# In[29]:


# og
output_mat = conv2d_to_mat(input_shape, weight_tensor, bias, stride=stride, padding=padding)
print("Dense Matrix (Original Function):")
print(output_mat, output_mat.shape)

output_sparse = conv2d_to_sparse(input_shape, weight_tensor, bias, stride=stride, padding=padding)
print("\nSparse Matrix (Modified Function):")
print(output_sparse.toarray(), output_mat.shape)


# In[ ]:





# In[ ]:




