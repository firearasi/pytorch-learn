# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

with open('data.txt','r') as f:
    data_list=f.readlines()
    data_list=[f.split('\n')[0] for f in data_list]
    data_list=[f.split(',') for f in data_list]
    data_list=[[float(f[0]),float(f[1]),float(f[2])] for f in data_list]

x_data= [[d[0],d[1]] for d in data_list]
y_data=[d[2] for d in data_list]
class0 = [a for a in data_list if a[2]==0]
class1 = [a for a in data_list if a[2]==1]


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr=nn.Linear(2,1)
        self.sm=nn.Sigmoid()

    def forward(self, x):
        return self.sm(self.lr(x))

logistics_model=LogisticRegression()
criterion=nn.BCELoss()
optimizer = optim.SGD(logistics_model.parameters(), lr=1e-3, momentum=0.9)

x_data=torch.FloatTensor(x_data)
y_data=torch.FloatTensor(y_data)

x=Variable(x_data)
y=Variable(y_data)

for epoch in range(40000):
    out = logistics_model(x)
    loss = criterion(out,y)
    print_loss = loss.data[0]
    mask=out.ge(0.5).float()
    correct=(mask==y).sum()
    acc = correct.data[0]/x.size(0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%1000==0:
        print('*'*20)
        print('epoch {}'.format(epoch+1))
        print('loss {:.5f}'.format((print_loss)))
        print('acc {:.5f}'.format(acc))




plt.plot([c[0] for c in class0], [c[1] for c in class0], 'ro', label='c0')
plt.plot([c[0] for c in class1], [c[1] for c in class1], 'bo', label='c1')
w0, w1=logistics_model.lr.weight[0]
w0=w0.data[0]
w1=w1.data[0]
b=logistics_model.lr.bias.data[0]
line_x=np.arange(30,100,0.1)
line_y=(-w0*line_x-b)/w1
plt.plot(line_x,line_y, label='Fitted separation')

plt.legend(loc='best')
plt.title('Original labels')
plt.show()