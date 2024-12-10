import torch
from torch import nn, optim 

import numpy as np 
import matplotlib.pyplot as plt 
import math
import open3d as o3d
from tqdm import tqdm
import rff

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=4): 
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Network(nn.Module):
    def __init__(self, r_1,r_2,r_3, mid_channel, posdim):
        super(Network, self).__init__()
        self.posdim = posdim
        
        self.U_net = nn.Sequential(SineLayer(posdim, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_1))
        
        self.V_net = nn.Sequential(SineLayer(posdim, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_2))
        
        self.W_net = nn.Sequential(SineLayer(posdim, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_3))
        
        self.encoding = rff.layers.GaussianEncoding(1.0, 1, posdim//2) #sigma不能太大，b的选择是标准高斯分布

    def forward(self, centre, x, flag):
        x = torch.cat([self.encoding(x[:,i].unsqueeze(1)) for i in range(3)], dim=1)
        U = self.U_net(x[:, :self.posdim]) #[299, 59]
        V = self.V_net(x[:, self.posdim:2*self.posdim]) #[299, 59]
        W = self.W_net(x[:, 2 * self.posdim:]) #[299, 59]
        # U = self.U_net(x[:,0].unsqueeze(-1)) #[299, 59]
        # V = self.V_net(x[:,1].unsqueeze(-1)) #[299, 59]
        # W = self.W_net(x[:,2].unsqueeze(-1)) #[299, 59]
        if flag == 1:
            output = torch.einsum('abc,na,nb,nc -> n', centre, U, V, W)
        elif flag == 2:
            output = torch.einsum('abc,ia,jb,kc -> ijk', centre, U, V, W)
        return output

if __name__ == '__main__':
    data_all = ["data/heartp0.05"] 
    dtype = torch.cuda.FloatTensor
    #################
    # Here are the hyperparameters.
    lr_real = 0.000001 
    thres = 0.01
    down = 5 
    max_iter = 2001
    gamma_1 = 0.3
    gamma_2 = 0.3
    
    #################
    for data in data_all:
        pcd = o3d.io.read_point_cloud(data+ '.pcd')
        X_np = np.array(pcd.points)[:,:] + 5
        add_border = 0.1
        
        # 归一化点云数据
        min_vals = X_np.min(axis=0) - add_border
        max_vals = X_np.max(axis=0) + add_border
        X_np = (X_np - min_vals) / (max_vals - min_vals)
        
        n = X_np.shape[0]
        mid_channel = n
        r_1 = int(n/down)
        r_2 = int(n/down)
        r_3 = int(n/down)
        
        X_gt = torch.zeros(n,1).type(dtype)
        U_input = (torch.from_numpy(X_np[:,0])).reshape(n,1).type(dtype)
        U_input.requires_grad=True
        V_input = (torch.from_numpy(X_np[:,1])).reshape(n,1).type(dtype)
        V_input.requires_grad=True
        W_input = (torch.from_numpy(X_np[:,2])).reshape(n,1).type(dtype)
        W_input.requires_grad=True
        centre = torch.zeros(r_1,r_2,r_3).type(dtype)
        stdv = 1 / math.sqrt(centre.size(0))
        centre.data.uniform_(-stdv, stdv)
        centre.requires_grad=True
        x_input = torch.cat((U_input, V_input, W_input), dim=1)

        model = Network(r_1,r_2,r_3, mid_channel, posdim=128).type(dtype)
        params = []
        params += [x for x in model.parameters()]
        params += [centre]
        optimizier = optim.Adam(params, lr=lr_real) 
        rand_num = 30
        
        with tqdm(total=max_iter) as pbar:
            for iter in range(max_iter):
                U_random = (torch.min(U_input)-add_border + 
                            (torch.max(U_input)-torch.min(U_input)+2*add_border) * torch.rand(rand_num,1).type(dtype))
                V_random = (torch.min(V_input)-add_border + 
                            (torch.max(V_input)-torch.min(V_input)+2*add_border) * torch.rand(rand_num,1).type(dtype))
                W_random = (torch.min(W_input)-add_border + 
                            (torch.max(W_input)-torch.min(W_input)+2*add_border) * torch.rand(rand_num,1).type(dtype))
                x_random = torch.cat((U_random,V_random, W_random), dim=1)

                X_Out = model(centre, x_input, flag = 1) #[299,1,1]
                loss_1 = torch.norm(X_Out - X_gt, 1)
                X_Out_off = model(centre, x_random, flag = 2) #只用于计算SDF损失 [30,30,30]
                grad_ = gradient(X_Out_off, x_random) #shape [30, 3]
                loss_2 = gamma_1 * torch.norm(grad_.norm(dim=-1) - rand_num**2, 1) #norm函数默认是F范数
                loss_3 = gamma_2 * torch.norm(torch.exp(-torch.abs(X_Out_off)), 1) 
                loss = loss_1 + loss_2 + loss_3

                optimizier.zero_grad()
                loss.backward(retain_graph=True)
                optimizier.step()
                
                pbar.set_postfix({'loss_1': f"{loss_1:.4f}", 'loss_2': f"{loss_2:.4f}", 'loss_3': f"{loss_3:.4f}"})
                pbar.update()
                # continue
                if iter % 500 == 0:
                    print('iteration:', iter)
                    number = 60
                    range_ = torch.from_numpy(np.array(range(number))).type(dtype)
                    u = (torch.min(U_input)-add_border + (
                        torch.max(U_input)-torch.min(U_input)+2*add_border) * (range_/number)).reshape(number,1)
                    v = (torch.min(V_input)-add_border + (
                        torch.max(V_input)-torch.min(V_input)+2*add_border) * (range_/number)).reshape(number,1)
                    w = (torch.min(W_input)-add_border + (
                        torch.max(W_input)-torch.min(W_input)+2*add_border) * (range_/number)).reshape(number,1)
                    x_in = torch.cat((u,v,w),dim=1)
                    out = model(centre,x_in, flag = 2).detach().cpu().clone()
                    idx = (torch.where(torch.abs(out)<thres))
                    Pts = torch.cat((u[idx[0]],v[idx[1]]),dim = 1)
                    Pts = torch.cat((Pts,w[idx[2]]),dim = 1).detach().cpu().clone().numpy()
                    
                    size_pc = 6
                    fig = plt.figure(figsize=(15,30))
                    ax = plt.subplot(121, projection='3d')
                    xs = Pts[:,0]
                    ys = Pts[:,1]
                    zs = Pts[:,2]
                    ax.scatter(xs, ys, zs,s=size_pc)
                    ax.view_init(elev=30, azim=90)
                    
                    ax = fig.add_subplot(122, projection='3d')
                    xs = X_np[:,0]
                    ys = X_np[:,1]
                    zs = X_np[:,2]
                    ax.scatter(xs, ys, zs,s=size_pc)
                    ax.view_init(elev=30, azim=90)
                    plt.show()