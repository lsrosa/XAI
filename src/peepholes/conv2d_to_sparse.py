import numpy as np
import torch
from warnings import warn

def conv2d_to_sparse(input_shape, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1,1), device='cpu', verbose=False, warns=True):

    if dilation != (1,1):
        raise RuntimeError('This functions does not account for dilation, if you extendent it, please send us a PR ;).')
    
    if padding != (0, 0):
        input_shape = input_shape[0:1] + torch.Size([x+2*y for x, y in zip(input_shape[-2:], padding)])
        if warns: warn('Do not forget to pad your input accoding to the Conv2d padding. Deactivate this warning passing warns=False as argument.', stacklevel=2)

    Cin, Hin, Win = input_shape
    Cout = weight.shape[0]
    Hk = weight.shape[2]
    Wk = weight.shape[3]
    kernel = weight

    Hout = int(np.floor((Hin - dilation[0]*(Hk - 1) -1)/stride[0] + 1))
    Wout = int(np.floor((Win - dilation[1]*(Wk - 1) -1)/stride[1] + 1))

    shape_out = torch.Size((Cout*Hout*Wout, Cin*Hin*Win+1))
    
    crow = (torch.linspace(0, shape_out[0], shape_out[0]+1)*(Hk*Wk*Cin+1)).int()
    nnz = crow[-1]
    
    # getting columns
    cols = torch.zeros(Cout*Hout*Wout, Cin*Hk*Wk+1, dtype=torch.int)
    data = torch.zeros(Cout*Hout*Wout, Cin*Hk*Wk+1)
    
    base_row = torch.zeros(Cin*Hk*Wk, dtype=torch.int)
    for cin in range(Cin):
        c_shift = cin*(Hin*Win)
        for hk in range(Hk):
            h_shift = hk*Win
            for wk in range(Wk):
                idx = cin*Hk*Wk+hk*Wk+wk
                w_shift = wk
                base_row[idx] = c_shift+h_shift+w_shift
        
    for cout in range(Cout): 
        k = kernel[cout]
        _d = torch.hstack((k.flatten(), bias[cout]))
        for ho in range(Hout):
            h_shift = ho*Win*stride[0]
            for wo in range(Wout):
                w_shift = wo*stride[1]
                idx = cout*Hout*Wout+ho*Wout+wo
                shift = h_shift+w_shift
                cols[idx,:-1] = base_row+shift
                data[idx] = _d

    # add bias as the last column                    
    cols[:,-1] = Cin*Hin*Win

    cols = cols.flatten()
    data = data.flatten()

    csr_mat = torch.sparse_csr_tensor(crow, cols, data, size=shape_out, device=device)
    return csr_mat 











