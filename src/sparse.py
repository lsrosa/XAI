import torch
from models.conv2d_to_sparse import conv2d_to_sparse as c2s
from scipy.sparse.linalg import svds as svdnp
import numpy as np
import scipy.linalg as linalg
import scipy

if __name__ == '__main__':
    c = torch.nn.Conv2d(2, 3, 3)
    w = c.weight
    wnp = w.detach().numpy()
    b = c.bias
    bnp = b.detach().numpy()

    x = torch.rand(2, 5, 5)
    xnp = x.detach().numpy()
    r = c(x)
    print('conv r: ', r)

    csrnp = c2s(x.shape, w, b) 
    
    '''
    s = 1000 
    a = torch.zeros(s, s)
    b = torch.rand(s, s)
    
    # populate some values of matrix a
    a[b<0.1] = b[b<0.1]
    #print('input matrix', a)

    # numpy version of matrix a
    anp = a.detach().numpy()

    csrnp = scipy.sparse.csr_matrix(anp)
    #print('numpy csr: ', csrnp)
    snp,vnp,dnp = svdnp(csrnp, k=s-1)
    vnp = np.diag(vnp)
    rnp = snp@vnp@dnp
    print('numpy reconstruction error: ', np.linalg.norm(rnp-anp))

    csr = a.to_sparse_csr()
    #print('torch csr: ', csr)
    s, v, d = torch.svd_lowrank(csr, q=s-1)
    v = torch.diag(v)
    r = s@v@d.T
    print('torch reconstruction: ', torch.norm(r-a))
    '''
