import torch
from models.conv2d_to_sparse import conv2d_to_sparse as c2s
from models.conv2d_to_sparse import conv2d_to_sparse2 as c2s2
from scipy.sparse.linalg import svds as svdnp
import numpy as np
import scipy.linalg as linalg
import scipy
from torch.nn.functional import conv2d
from torch.nn import Parameter as P
from time import time
from numpy.random import randint as ri

if __name__ == '__main__':
    nc = ri(2, 20) # n channels
    kw = ri(2, 10) # kernel width 
    kh = ri(2, 10) # kernel height
    iw = ri(5, 50) # image width
    ih = ri(5, 50) # image height
    ns = 1 # n samples
    cic = nc # conv in channels
    coc = ri(2, 20) # conv out channels
    sh = ri(2 ,10)
    sw = ri(2, 10)
    
    print('cic, coc: ', cic, coc)
    print('kernel h, w: ', kh, kw)
    print('image h, w: ', ih, iw)
    print('strid h, w: ', sh, sw)
    
    c = torch.nn.Conv2d(cic, coc, (kh, kw), stride=(sh, sw))

    #_w = torch.tensor([float(i) for i in range(1,1+nc*coc*kw*kh)]).reshape(coc, nc, kw, kh)
    #c.weight = P(_w, requires_grad=True)
    
    w = c.weight
    wnp = w.detach().numpy()

    b = c.bias
    bnp = b.detach().numpy()
    #print('kernels, bias', w, b)

    x = torch.rand(ns, nc, ih, iw)
    xnp = x.detach().numpy()
    r = c(x)
    #print('conv r: ', r, r.shape)

    t0 = time()
    csrnp, csr = c2s(x.shape[1:], w, b, stride=c.stride, padding=c.padding) 
    t_prev = time()-t0
    #print('csrnp: ', torch.tensor(csrnp.todense()), csrnp.todense().shape)
    xunp = np.hstack((xnp.flatten(), np.ones(1))) 
    lcnp = csrnp.todense()
    runp = lcnp@xunp
    print('error runp: ', (r-torch.tensor(runp).reshape(r.shape)).sum())
    #print('shape np: ', lcnp.shape)

    #print('csr: ', csr.to_dense(), csr.to_dense().shape)
    xu = torch.hstack((x.flatten(), torch.ones(1)))
    lc = csr.to_dense()
    ru = lc@xu
    print('error ru: ', (r-ru.reshape(r.shape)).sum())
    #print('\ncsr: ', lc)
    
    '''
    aa = torch.tensor([[i for i in range(lc.shape[1])]]).repeat(lc.shape[0],1)
    aa = (aa[lc!=0]).reshape(lc.shape[0], cic*kh*kw+1)
    print('debug: ', aa)
    '''

    t0 = time()
    my_csr = c2s2(x.shape[1:], w, b, stride=c.stride, padding=c.padding)
    t_curr = time()-t0
    #print('my csr: ', my_csr.to_dense(), my_csr.to_dense().shape)
    lc = my_csr.to_dense()
    ru = lc@xu
    #print('my ru: ', ru.reshape(r.shape))
    #print('\ncsr: ', lc)
    print('error ru2: ', (r-ru.reshape(r.shape)).sum())
    print('shape to: ', lc.shape, lc.dtype)
    print('time prev, curr: ', t_prev, t_curr) 
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
