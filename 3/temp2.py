# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import numpy as np
a = torch.Tensor([[2,3],[4,8],[7,9]])
print('a is :{}'.format(a))
print('a size is {}'.format(a.size()))

b=torch.LongTensor([[2,2],[3,3],[4,4]])
print('b is: {}'.format(b))

c=torch.zeros((3,2))
c
d=torch.randn((4,3))
d
a[0,1]=20
a
numpy_b=b.numpy()
b
numpy_b

e=np.array([[1,2],[3,5]])
torch_e=torch.from_numpy(e)
torch_e
f_torch_e=torch_e.float()
f_torch_e
f_torch_e.cuda()
