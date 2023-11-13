import numpy as np
import torch
from torch.utils import data   
from torch import nn

def gen_data(w,b,example_nums):
    X = torch.normal(0,1,(example_nums,len(w)))
    y = torch.matmul(X,w)+b
    y += torch.normal(0,0.01,y.shape)
    return X,torch.reshape(y,(-1,1))

def load_array(data_arrarys,batch_size,is_train = True):
    dataset = data.TensorDataset(*data_arrarys)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

def linear_reg():
    real_w = torch.tensor([2,-3.4])
    real_b = 4.2
    feature,labels = gen_data(real_w,real_b,1000)
    batch_size = 10
    data_iter = load_array((feature, labels), batch_size)
    net = nn.Sequential(nn.Linear(2,1))
    net[0].weight.data.normal_(0,0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X,y in data_iter:
            l = loss(net(X),y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(feature),labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

if  __name__  == '__main__':
    linear_reg()

