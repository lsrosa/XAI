import torch
from models.conv2d_to_sparse import conv2d_to_sparse as c2s
from models.conv2d_to_sparse import conv2d_to_sparse2 as c2s2
from scipy.sparse.linalg import svds as svdnp
import numpy as np
import scipy.linalg as linalg
import scipy
from torch.nn.functional import conv2d
from torch.nn import Parameter as P

if __name__ == '__main__':
    nc = 3 # n channels
    kw = 2 # kernel width 
    kh = 3 # kernel heigh 
    iw = 4 # image width
    ih = 4 # image height
    ns = 1 # n samples
    cic = nc # conv in channels
    coc = 2 # conv out channels

    c = torch.nn.Conv2d(cic, coc, (kh, kw))
    _w = torch.tensor([float(i) for i in range(1,1+nc*coc*kw*kh)]).reshape(coc, nc, kw, kh)
    c.weight = P(_w, requires_grad=True)
    
    w = c.weight
    wnp = w.detach().numpy()

    b = c.bias
    bnp = b.detach().numpy()
    print('kernels, bias', w, b)

    x = torch.rand(ns, nc, ih, iw)
    print('input shape: ', x.shape)
    xnp = x.detach().numpy()
    r = c(x)
    print('conv r: ', r, r.shape)

    csrnp, csr = c2s(x.shape[1:], w, b, stride=c.stride, padding=c.padding) 
    #print('csrnp: ', torch.tensor(csrnp.todense()), csrnp.todense().shape)
    xunp = np.hstack((xnp.flatten(), np.ones(1))) 
    lcnp = csrnp.todense()
    runp = lcnp@xunp
    print('runp: ', torch.tensor(runp).reshape(r.shape))
    
    #print('csr: ', csr.to_dense(), csr.to_dense().shape)
    xu = torch.hstack((x.flatten(), torch.ones(1)))
    lc = csr.to_dense()
    ru = lc@xu
    print('ru: ', ru.reshape(r.shape))
    
    my_csr = c2s2(x.shape[1:], w, b, stride=c.stride, padding=c.padding)
    #print('my csr: ', my_csr.to_dense(), my_csr.to_dense().shape)
    lc = my_csr.to_dense()
    ru = lc@xu
    print('my ru: ', ru.reshape(r.shape))

    '''
    s = 1000 
    a = torch.zeros(s, s)
    b = torch.rand(s, s)
    
    # populate some values of matrix a
    a[b<0.1] = b[b<0.1]
    #print('input matrix', a)

    anp = a.detach().numpy()
    # numpy version of matrix a

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
