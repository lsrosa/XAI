from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
import torch
import sys
from torch.utils.data import DataLoader

if __name__ == '__main__':
    n = 10
    bs = 2
    device = 'cpu' 
    n_dicts = 2 
    
    # seems like batch size is really important
    td = TensorDict(batch_size=n, device=device)
    for j in range(n_dicts):
        td['a%d'%j] = MMT.empty(shape=(n,5))
         
    for i in range(n):
        d = {}
        # create data
        for j in range(2):
            d['a%d'%j] = torch.rand((1, 5))
        td[i] = TensorDict(d) 

    print(sys.getsizeof(td))
    td.memmap("./banana")
    del td

    td2 = TensorDict.load_memmap('./banana')
    print('td2:', td2)
    for e in td2.keys():
        print(td2[e])    

    dl = DataLoader(dataset=td2, batch_size=3, collate_fn=lambda x: x)
    print('\n dataloader: ', len(dl.dataset))
    for d in dl:
        print('batch')
        for i in range(n_dicts):
            print(d['a%d'%i].contiguous())
     
