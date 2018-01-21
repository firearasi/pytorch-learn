# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 09:17:46 2017

@author: cheng
"""

import torch
from torch.autograd import Variable
a=torch.rand(5,4)
a.size()

x=Variable(torch.Tensor([3]), requires_grad = True)
y=Variable(torch.Tensor([5]), requires_grad = True)

z = x * y
u = z ** 2
u.backward()

z.grad.data

x.grad.data
y.grad.data
