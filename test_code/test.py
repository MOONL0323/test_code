import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

class PowerReLU(nn.Module):

    def __init__(self, inplace=False, power=3):
        super(PowerReLU, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input):
        y = F.relu(input, inplace=self.inplace)
        return torch.pow(y, self.power)

class Block(nn.Module):
    """
        这是一个Block模块，包括两个全连接层和一个激活函数
        前向传播采取的是y = phi(L2(phi(L1(x)))) + x
        其中的x其实是利用了残差网络，保留了原始的输入
    """
    def __init__(self, in_N, width, out_N, phi=PowerReLU()):
        super(Block, self).__init__()
        # create the necessary linear layers
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        # choose appropriate activation function
        self.phi = nn.Tanh()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x

class drrnn(nn.Module):
    """
        这是一个drrnn模块，包括一个输入层，一个输出层，中间是由多个Block模块组成的堆叠 
        参数：
        in_N -- 输入维度
        out_N -- 输出尺寸
        m -- 形成块的层的宽度
        depth -- 要堆叠的块数
        phi -- 激活函数

        该模块的神经网络结构如下：
        x -> L1 -> L2 -> phi -> L1 -> L2 -> phi -> ... -> L1 -> L2 -> phi -> L1 -> L2 -> y
        包含一个线性输入层，deepth=4个Block模块，一个线性输出层
        
    """

    def __init__(self, in_N, m, out_N, depth=5, phi=PowerReLU()):
        super(drrnn, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.Tanh()
        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N, m))

        # add middle blocks to list
        for i in range(depth):
            self.stack.append(Block(m, m, m))

        # add output linear layer
        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        # first layer
        # 向前传播
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def train( x_train, y_train, z_train,x,y, epochs=50000, lr=1e-4):
    epochs = 50000
    in_N = 2
    m = 50
    out_N = 1
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = drrnn(in_N, m, out_N).to(device)
    model.apply(weights_init)
    criterion = nn.MSELoss()
    losses = []
    best_loss = 1000
    best_epoch = 0
    for epoch in range(epochs):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()
        #x,y,z转化为tensor
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        z_train = torch.tensor(z_train, dtype=torch.float32)

        #x转化为列向量，y转化为列向量，并且拼接在一起
        inputs = torch.cat((x_train.view(-1,1), y_train.view(-1,1)), 1)
        targets = z_train.reshape(-1,1)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if torch.abs(loss) < best_loss:
                  best_loss = torch.abs(loss).item()
                  best_epoch = epoch
                  model_path = torch.save(model, 'model.mdl')
        if epoch % 1000 == 0:
            print('Epoch %d, Loss: %.6f' % (epoch, loss.item()))

        losses.append(loss.item())
        if loss.item() < 1e-5 or epoch == epochs:
            with torch.no_grad():
                plt.figure(1)
                inputs = torch.tensor(np.column_stack((x_train.flatten(), y_train.flatten())), dtype=torch.float32)
                outputs = model(inputs)
                z_pred = outputs.reshape(x_train.shape)
                plt.contour(x_train, y_train, z_pred,20)
                plt.savefig('final.eps')
                plt.show()
                plt.close()

                plt.figure(2)
                times = np.arange(epoch)
                plt.semilogy(times, losses, label='loss')
                plt.xlabel('epoch')
                plt.ylabel('log(loss)')
                plt.savefig('loss-epoch.eps')
                plt.show()
                break
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    inputs = torch.cat((x.view(-1,1), y.view(-1,1)), 1)
    z = model(inputs)
    z = z.reshape(x.shape)
    print(z)
    return losses,model_path,best_epoch,best_loss,z





