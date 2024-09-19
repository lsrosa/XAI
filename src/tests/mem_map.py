from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
import torch

if __name__ == '__main__':
    mmt = MMT.empty(shape=(5,5), filename='./banana', existsok=True)
    print(mmt, mmt.dtype, mmt.shape)

    for i in range(5):
        mmt[i] = torch.rand((1, 5))
        print(mmt)

    mmt2 = MMT.from_filename(dtype=mmt.dtype, shape=mmt.shape, filename='./banana')
    print('mmt2:', mmt2)

