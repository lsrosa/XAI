import tqdm
from scipy.sparse import csr_matrix, coo_matrix
import scipy
import numpy as np
from tqdm import tqdm
from itertools import product
import torch

def conv2d_to_sparse(input_shape, weight_tensor, bias, stride=(1, 1), padding=(0, 0)):

    Cin, Hin, Win = input_shape
    Cout = weight_tensor.shape[0]
    kernel_shape = tuple(weight_tensor.shape[2:])
    kernel = weight_tensor.detach().numpy()
    bias = bias.detach().numpy()

    Hout = int(np.floor((Hin + 2*padding[0] - (kernel_shape[0] - 1) -1)/stride[0] + 1))
    Wout = int(np.floor((Win + 2*padding[1] - (kernel_shape[1] - 1) -1)/stride[1] + 1))

    rows = []
    cols = []
    data = []

    for f, i, j, c in tqdm(product(range(Cout), range(Hout), range(Wout), range(Cin))):
        row = f * Hout * Wout + i * Wout + j 
        col_start = c * Hin * Win 
        weight = kernel[f, c]
        
        for m in range(kernel_shape[0]):
            for n in range(kernel_shape[1]):
                col = col_start + ((m * Win + n + i * stride[1] * Win + j * stride[0]) % (Win * Hin))

                rows.append(row)
                cols.append(col)
                data.append(weight[m, n])
    
    # add bias as the last column
    for f in tqdm(range(Cout)):
        for i in range(Hout):
            for j in range(Wout):
                row = f * Hout * Wout + i * Wout + j
                rows.append(row)
                cols.append(Cin * Hin * Win)
                data.append(bias[f])
    
    data, rows, cols = np.array(data), np.array(rows), np.array(cols)

    # convert to csr
    m = csr_matrix((data, (rows, cols)), shape=(Cout * Hout * Wout, Cin * Hin * Win + 1))
    csr_mat = torch.sparse_csr_tensor(m.indptr, m.indices, m.data, size=m.shape)
    return m, csr_mat

def conv2d_to_sparse2(input_shape, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1,1)):

    Cin, Hin, Win = input_shape
    #print('input shape: ', input_shape, Cin, Hin, Win) 
    Cout = weight.shape[0]
    #print('Cout: ', Cout)
    Hk = weight.shape[2]
    Wk = weight.shape[3]
    #print('kernel shape: ', Hk, Wk)
    kernel = weight
    #print('kernel: ', kernel)

    Hout = int(np.floor((Hin + 2*padding[0] - dilation[0]*(Hk - 1) -1)/stride[0] + 1))
    Wout = int(np.floor((Win + 2*padding[1] - dilation[1]*(Wk - 1) -1)/stride[1] + 1))
    #print('hout, wout: ', Hout, Wout)
    
    shape_out = torch.Size((Cout*Hout*Wout, Cin*Hin*Win+1))
    print('shape out: ', shape_out)
    
    crow = (torch.linspace(0, shape_out[0], shape_out[0]+1)*(Hk*Wk*Cin+1)).int()
    print('crow: ', crow)
    nnz = crow[-1]
    print('nnz: ', nnz, Cout*Hout*Wout)
    
    # getting columns
    cols = []
    for f, i, j, c in tqdm(product(range(Cout), range(Hout), range(Wout), range(Cin))):
        col_start = c * Hin * Win 
        
        for m in range(Hk):
            for n in range(Wk):
                col = col_start + ((m * Win + n + i * stride[1] * Win + j * stride[0]) % (Win * Hin))
                cols.append(col)
    
    # add bias as the last column                    
    for f in tqdm(range(Cout)):
        for i in range(Hout):
            for j in range(Wout):
                cols.append(Cin * Hin * Win)
    
    cols = torch.tensor(cols)

    # data is the kernel values, plus bias
    data = torch.zeros(Cout*Hout*Cout, Cin*Hk*Wk+1)
    for cout in range(Cout):
        k = kernel[cout]
        _d = torch.hstack((k.flatten(), bias[cout]))
        for i, hout in enumerate(range(Hout*Wout)):
            data[cout*Hout*Wout+i] = _d
    data = data.flatten()

    csr_mat = torch.sparse_csr_tensor(crow, cols, data, size=shape_out)
    return csr_mat 
