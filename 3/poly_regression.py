# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from torch import nn,optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


def make_features(x):
    x.unsqueeze_(1)
    x0=x**0
    x1=x
    x2=x**2
    x3=x**3
    out = torch.cat([x0,x1,x2,x3],dim=1)
    return out

W_target = torch.FloatTensor([0.9, 0.5,3,2.4]).unsqueeze(1)

def f(x):
    return x.mm(W_target)
#f(make_features(x))

# if dim(x)=m*4, then dim(y)=m*1
def get_batch(batch_size=32):
    random=torch.randn(batch_size)
    x=make_features(random)
    y=f(x)    
    
    return Variable(x), Variable(y)

class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly=nn.Linear(4,1)
    
    def forward(self,x):
        out=self.poly(x)
        return out
    
    
model = poly_model()
optimizer=optim.SGD(model.parameters(), lr=1e-3)
criterion=nn.MSELoss()

epoch = 0
while True:
    epoch = epoch + 1
    batch_x, batch_y = get_batch()
    output = model(batch_x)
    loss=criterion(output, batch_y)
    print_loss=loss.data[0]
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch%20==0:
        print('Epoch {}. Loss {:.6f}'.format(epoch, print_loss))
    
    if print_loss<1:
        break

model.eval()

eval_x, eval_y = get_batch(120)
output = model(eval_x)

#output_y-eval_y
eval_x = eval_x[:,1]

x=eval_x.data.numpy()
y=eval_y.data.squeeze(1).numpy()

sorting=np.argsort(x)

plt.plot(x[sorting], y[sorting], label = 'Original data', color = 'r')

output=output.data.squeeze(1).numpy()

plt.plot(x[sorting],output[sorting], label='Fitted line', color = 'b')
plt.legend()
plt.show()
