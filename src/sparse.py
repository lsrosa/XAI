import torch
from models.conv2d_to_sparse import conv2d_to_sparse as c2s
from time import time
from numpy.random import randint as ri

if __name__ == '__main__':
    for i in range(1):
        nc = ri(2, 20) # n channels
        kw = ri(2, 10) # kernel width 
        kh = ri(2, 10) # kernel height
        iw = ri(10, 50) # image width
        ih = ri(10, 50) # image height
        ns = 1 # n samples
        cic = nc # conv in channels
        coc = ri(2, 20) # conv out channels
        sh = ri(2 ,10)
        sw = ri(2, 10)
        
        print('\n-------------------------')
        print('cic, coc: ', cic, coc)
        print('kernel h, w: ', kh, kw)
        print('image h, w: ', ih, iw)
        print('strid h, w: ', sh, sw)
        
        c = torch.nn.Conv2d(cic, coc, (kh, kw), stride=(sh, sw), dilation=(1,1), padding=(1,1))
        w = c.weight
        b = c.bias
        
        x = torch.rand(ns, nc, ih, iw)
        r = c(x)

        t0 = time()
        my_csr = c2s(x[0].shape, w, b, stride=c.stride, padding=c.padding, dilation=c.dilation)
        '''
        print('SVDing')
        s, v, d = torch.svd_lowrank(my_csr, q=10)
        print('v:',  v)
        print('SVDone')
        '''

        t_curr = time()-t0
        lc = my_csr.to_dense()
        xu = torch.hstack((x.flatten(), torch.ones(1)))
        ru = lc@xu
        error = torch.norm(r-ru.reshape(r.shape))/torch.norm(r) 
        print('error ru: ', error)
        print('time: ', t_curr) 
        if error > 1.0:
            raise RuntimeError('Girl, go debug that conv.')
        print('-------------------------\n')
