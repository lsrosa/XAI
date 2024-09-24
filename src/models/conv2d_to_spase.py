import tqdm
from scipy.sparse import csr_matrix, coo_matrix
import scipy

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
    
    Hout = int(np.floor((Hin + 2*padding[-2] - (kernel_shape[-2] - 1) -1)/stride[-2] + 1))
    Wout = int(np.floor((Win + 2*padding[-1] - (kernel_shape[-1] - 1) -1)/stride[-1] + 1))
    print(Hout, Wout)
    
    rows = []
    cols = []
    data = []
    
    for f, i, j, c in tqdm(product(range(Cout), range(Hout), range(Wout), range(Cin))):
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
    for f in tqdm(range(Cout)):
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


