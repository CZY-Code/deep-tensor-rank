# We think latent code contain the rank vector {R_1, R_2, R_3} of Tucker Decomposition
import torch
from torch import nn, optim 
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import math
import open3d as o3d
from tqdm import tqdm

from Demo_PoinCloud_upsampling import SineLayer
from utils.deep_sdf_decoder import Decoder


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    #当SDF输出的梯度为1的时候
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

#################
# Here are the hyperparameters.
dataPath = "data/heartp0.05"
lr_real = 0.001 
thres = 0.001
down = 5 
max_iter = 2001
omega = 4

pcd = o3d.io.read_point_cloud(dataPath + '.pcd')
data_np = np.array(pcd.points)[:,:] #n个三维坐标
n = data_np.shape[0] #[299,3]
mid_channel = 256
latent_channel = 128
CPrank = 64

r_1 = int(n/down)
r_2 = int(n/down)
r_3 = int(n/down)
X_gt = torch.zeros(n, 1).type(dtype) #SDF中的物体零平面
#################

class LatentNet(nn.Module):
    def __init__(self):
        super(LatentNet, self).__init__()
        self.latent_x = nn.Parameter(torch.normal(0, 0.1, size=(1, latent_channel)).type(dtype))
        self.latent_y = nn.Parameter(torch.normal(0, 0.1, size=(1, latent_channel)).type(dtype))
        self.latent_z = nn.Parameter(torch.normal(0, 0.1, size=(1, latent_channel)).type(dtype))
        self.CPdiag =  nn.Parameter(torch.ones(CPrank).type(dtype))
        stdv = 1 / math.sqrt(self.CPdiag.size(0))
        self.CPdiag.data.uniform_(-stdv, stdv)

        self.latent = nn.Embedding(1, 128)

        self.LU_net = nn.Sequential(nn.Linear(128, mid_channel, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(mid_channel, mid_channel, bias=False),
                                    nn.Dropout(p=0.1))
        self.LV_net = nn.Sequential(nn.Linear(128, mid_channel, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(mid_channel, mid_channel, bias=False),
                                    nn.Dropout(p=0.1))
        self.LW_net = nn.Sequential(nn.Linear(128, mid_channel, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(mid_channel, mid_channel, bias=False),
                                    nn.Dropout(p=0.1))
        
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True))
        self.V_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True))
        self.W_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True))
        
        self.SU_net = nn.Sequential(SineLayer(2*mid_channel, mid_channel, is_first=True),
                                    nn.Linear(mid_channel, CPrank))
        self.SV_net = nn.Sequential(SineLayer(2*mid_channel, mid_channel, is_first=True),
                                    nn.Linear(mid_channel, CPrank))
        self.SW_net = nn.Sequential(SineLayer(2*mid_channel, mid_channel, is_first=True),
                                    nn.Linear(mid_channel, CPrank))

    def forward(self, XYZaxis, form='Tucker'):
        X_axis = XYZaxis[:,0].unsqueeze(-1)
        Y_axis = XYZaxis[:,1].unsqueeze(-1)
        Z_axis = XYZaxis[:,2].unsqueeze(-1)
        U = self.U_net(X_axis) #[n_points, CPrank]
        V = self.V_net(Y_axis)
        W = self.W_net(Z_axis)
        
        m = X_axis.shape[0]
        LU = self.LU_net(self.latent_x.expand(m, -1)) #[n_points, CPrank]
        LV = self.LV_net(self.latent_y.expand(m, -1))
        LW = self.LW_net(self.latent_z.expand(m, -1))

        X_inputs = torch.cat([U, LU], dim=-1)
        Y_inputs = torch.cat([V, LV], dim=-1)
        Z_inputs = torch.cat([W, LW], dim=-1)
        SU = self.SU_net(X_inputs)
        SV = self.SV_net(Y_inputs)
        SW = self.SW_net(Z_inputs)

        centre = torch.zeros((CPrank, CPrank, CPrank)).type(dtype)
        for i in range(CPrank):
            centre[i, i, i] = self.CPdiag[i]
        if form == 'Tucker':
            output = torch.einsum('abc,ia,jb,kc -> ijk', centre, SU, SV, SW)
        elif form == 'pointWise': #把n提出来，就变成了三个向量和core的组合，空间维度为1*1*1，计算n次
            output = torch.einsum('abc,na,nb,nc -> n', centre, SU, SV, SW)
        else:
            raise NotImplementedError
        return output
        

latentModel = LatentNet().cuda()
#downsampling for learning latent code#
def trainLatentCode():
    iterNum4L = 1000
    gamma_1 = 0.01
    gamma_2 = 5

    params = []
    params += [x for x in latentModel.parameters()]
    optimizer = optim.Adam(params, lr=lr_real)

    rand_num = 30 #
    add_border = 0.1

    with tqdm(total=iterNum4L) as pbar:
        for iter in range(iterNum4L):
            #空间位置上的0.5倍稀疏采样
            m = int(n*0.5)
            sampled_data = data_np[np.random.choice(n, size=m, replace=False)]
            X_axis = (torch.from_numpy(sampled_data[:,0])).reshape(m,1).type(dtype)
            # X_axis.requires_grad=True
            Y_axis = (torch.from_numpy(sampled_data[:,1])).reshape(m,1).type(dtype)
            # Y_axis.requires_grad=True
            Z_axis = (torch.from_numpy(sampled_data[:,2])).reshape(m,1).type(dtype)
            # Z_axis.requires_grad=True
            XYZaxis = torch.cat((X_axis, Y_axis, Z_axis), dim=1)

            X_Out = latentModel(XYZaxis, form='pointWise')
            loss_1 = torch.norm(X_Out - X_gt, 1) #和全零gt的L1距离

            #在R^3空间上随机采样rand_num^3个点的坐标
            X_random = torch.from_numpy(np.min(sampled_data[:,0]) - add_border + 
                                        (np.max(sampled_data[:,0]) - np.min(sampled_data[:,0]) + 2 * add_border)
                                        * np.random.rand(rand_num, 1)).type(dtype)
            X_random.requires_grad = True
            Y_random = torch.from_numpy(np.min(sampled_data[:,1]) - add_border + 
                                        (np.max(sampled_data[:,1]) - np.min(sampled_data[:,1]) + 2 * add_border) 
                                        * np.random.rand(rand_num, 1)).type(dtype)
            Y_random.requires_grad = True
            Z_random = torch.from_numpy(np.min(sampled_data[:,2]) - add_border + 
                                        (np.max(sampled_data[:,2]) - np.min(sampled_data[:,2]) + 2 * add_border) 
                                        * np.random.rand(rand_num, 1)).type(dtype)
            Z_random.requires_grad = True
            axis_random = torch.cat((X_random, Y_random, Z_random), dim=1) #[num, 3]在三个轴上的位置随机采样
            #X_Out_off [rand_num,rand_num,rand_num]用于计算SDF损失
            X_Out_off = latentModel(axis_random, form='Tucker')
            grad_ = gradient(X_Out_off, axis_random) #[rand_num, 3]
            loss_2 = gamma_1 * torch.norm(grad_.norm(dim=-1) - rand_num**2, 1) #norm函数默认是F范数，积分变L1范数
            loss_3 = gamma_2 * torch.norm(torch.exp(-torch.abs(X_Out_off)), 1)

            loss = loss_1 + loss_2 + loss_3
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            pbar.set_postfix({'loss_1': f"{loss_1:.4f}", 'loss_2': f"{loss_2:.4f}", 'loss_3': f"{loss_3:.4f}"})
            pbar.update()
            # continue
            if iter % (iterNum4L-1) == 0 and iter != 0:
                print('iteration:', iter)
                number = 60
                range_ = torch.from_numpy(np.array(range(number))).type(dtype)
                u = (torch.min(X_axis)-add_border + (
                    torch.max(X_axis)-torch.min(X_axis)+2*add_border) * (range_/number)).reshape(number,1)
                v = (torch.min(Y_axis)-add_border + (
                    torch.max(Y_axis)-torch.min(Y_axis)+2*add_border) * (range_/number)).reshape(number,1)
                w = (torch.min(Z_axis)-add_border + (
                    torch.max(Z_axis)-torch.min(Z_axis)+2*add_border) * (range_/number)).reshape(number,1)
                x_in = torch.cat((u,v,w), dim=1)

                out = latentModel(x_in, form='Tucker').detach().cpu().clone()
                idx = (torch.where(torch.abs(out)<thres))
                Pts = torch.cat((u[idx[0]], v[idx[1]]), dim = 1)
                Pts = torch.cat((Pts, w[idx[2]]), dim = 1).detach().cpu().clone().numpy()
                
                size_pc = 6
                fig = plt.figure(figsize=(15, 30))
                ax = plt.subplot(121, projection='3d')
                xs = Pts[:,0]
                ys = Pts[:,1]
                zs = Pts[:,2]
                ax.scatter(xs, ys, zs,s=size_pc)
                ax.view_init(elev=30, azim=90)
                
                ax = fig.add_subplot(122, projection='3d')
                xs = sampled_data[:,0]
                ys = sampled_data[:,1]
                zs = sampled_data[:,2]
                ax.scatter(xs, ys, zs,s=size_pc)
                ax.view_init(elev=30, azim=90)
                plt.show()


def findLatentCode():
    iterNum4F = 1000
    gamma_1 = 0.01
    gamma_2 = 5
    optimizer = optim.Adam([latentModel.latent_x, latentModel.latent_y, latentModel.latent_z], lr=lr_real)
    rand_num = 30 #
    add_border = 0.1
    with tqdm(total=iterNum4F) as pbar:
        for iter in range(iterNum4F):
            #空间位置上的1.0倍采样
            m = int(n*1.0)
            sampled_data = data_np[np.random.choice(n, size=m, replace=False)]
            X_axis = (torch.from_numpy(sampled_data[:,0])).reshape(m,1).type(dtype)
            # X_axis.requires_grad=True
            Y_axis = (torch.from_numpy(sampled_data[:,1])).reshape(m,1).type(dtype)
            # Y_axis.requires_grad=True
            Z_axis = (torch.from_numpy(sampled_data[:,2])).reshape(m,1).type(dtype)
            # Z_axis.requires_grad=True
            XYZaxis = torch.cat((X_axis, Y_axis, Z_axis), dim=1)

            X_Out = latentModel(XYZaxis, form='pointWise')
            loss_1 = torch.norm(X_Out - X_gt, 1) #和全零gt的L1距离

            #在R^3空间上随机采样rand_num^3个点的坐标
            X_random = torch.from_numpy(np.min(sampled_data[:,0]) - add_border + 
                                        (np.max(sampled_data[:,0]) - np.min(sampled_data[:,0]) + 2 * add_border)
                                        * np.random.rand(rand_num, 1)).type(dtype)
            X_random.requires_grad = True
            Y_random = torch.from_numpy(np.min(sampled_data[:,1]) - add_border + 
                                        (np.max(sampled_data[:,1]) - np.min(sampled_data[:,1]) + 2 * add_border) 
                                        * np.random.rand(rand_num, 1)).type(dtype)
            Y_random.requires_grad = True
            Z_random = torch.from_numpy(np.min(sampled_data[:,2]) - add_border + 
                                        (np.max(sampled_data[:,2]) - np.min(sampled_data[:,2]) + 2 * add_border) 
                                        * np.random.rand(rand_num, 1)).type(dtype)
            Z_random.requires_grad = True
            axis_random = torch.cat((X_random, Y_random, Z_random), dim=1) #[num, 3]在三个轴上的位置随机采样
            #X_Out_off [rand_num,rand_num,rand_num]用于计算SDF损失
            X_Out_off = latentModel(axis_random, form='Tucker')
            grad_ = gradient(X_Out_off, axis_random) #[rand_num, 3]
            loss_2 = gamma_1 * torch.norm(grad_.norm(dim=-1) - rand_num**2, 1) #norm函数默认是F范数，积分变L1范数
            loss_3 = gamma_2 * torch.norm(torch.exp(-torch.abs(X_Out_off)), 1)

            loss = loss_1 + loss_2 + loss_3
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            pbar.set_postfix({'loss_1': f"{loss_1:.4f}", 'loss_2': f"{loss_2:.4f}", 'loss_3': f"{loss_3:.4f}"})
            pbar.update()
            # continue
            if iter % (iterNum4F-1) == 0 and iter != 0:
                print('iteration:', iter)
                number = 60
                range_ = torch.from_numpy(np.array(range(number))).type(dtype)
                u = (torch.min(X_axis)-add_border + (
                    torch.max(X_axis)-torch.min(X_axis)+2*add_border) * (range_/number)).reshape(number,1)
                v = (torch.min(Y_axis)-add_border + (
                    torch.max(Y_axis)-torch.min(Y_axis)+2*add_border) * (range_/number)).reshape(number,1)
                w = (torch.min(Z_axis)-add_border + (
                    torch.max(Z_axis)-torch.min(Z_axis)+2*add_border) * (range_/number)).reshape(number,1)
                x_in = torch.cat((u,v,w), dim=1)

                out = latentModel(x_in, form='Tucker').detach().cpu().clone()
                idx = (torch.where(torch.abs(out)<thres))
                Pts = torch.cat((u[idx[0]], v[idx[1]]), dim = 1)
                Pts = torch.cat((Pts, w[idx[2]]), dim = 1).detach().cpu().clone().numpy()
                
                size_pc = 6
                fig = plt.figure(figsize=(15, 30))
                ax = plt.subplot(121, projection='3d')
                xs = Pts[:,0]
                ys = Pts[:,1]
                zs = Pts[:,2]
                ax.scatter(xs, ys, zs,s=size_pc)
                ax.view_init(elev=30, azim=90)
                
                ax = fig.add_subplot(122, projection='3d')
                xs = sampled_data[:,0]
                ys = sampled_data[:,1]
                zs = sampled_data[:,2]
                ax.scatter(xs, ys, zs,s=size_pc)
                ax.view_init(elev=30, azim=90)
                plt.show()


def testLatentCode():
    gamma_1 = 0.3
    gamma_2 = 0.3
    r_1 = int(n/down)
    r_2 = int(n/down)
    r_3 = int(n/down)
    
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
    model = Network(r_1,r_2,r_3, mid_channel).type(dtype)
    params = []
    params += [x for x in model.parameters()]
    params += [centre]

    optimizier = optim.Adam(params, lr=lr_real) 
    rand_num = 30
    add_border = 0.1
    for iter in range(max_iter):
        U_random = (torch.min(U_input) - add_border + 
                    (torch.max(U_input) - torch.min(U_input) + 2 * add_border) * torch.rand(rand_num,1).type(dtype)) 
        V_random = (torch.min(V_input) - add_border + 
                    (torch.max(V_input) - torch.min(V_input) + 2 * add_border) * torch.rand(rand_num,1).type(dtype))
        W_random = (torch.min(W_input) - add_border + 
                    (torch.max(W_input) - torch.min(W_input) + 2 * add_border) * torch.rand(rand_num,1).type(dtype))
        x_random = torch.cat((U_random,V_random, W_random), dim=1)
        X_Out = model(centre, x_input, flag = 1)
        loss_1 = torch.norm((X_Out) - X_gt, 1)
        X_Out_off = model(centre, x_random, flag = 2) #只用于计算SDF损失
        grad_ = gradient(X_Out_off, x_random) #shape [30, 3]
        loss_2 = gamma_1 * torch.norm(grad_.norm(dim=-1) - rand_num**2, 1) #norm函数默认是F范数
        loss_3 = gamma_2 * torch.norm(torch.exp(-torch.abs(X_Out_off)), 1)
        loss = loss_1 + loss_2 + loss_3

        optimizier.zero_grad()
        loss.backward(retain_graph=True)
        optimizier.step()
        if iter % 200 == 0:
            print('iteration:', iter)
            number = 60
            range_ = torch.from_numpy(np.array(range(number))).type(dtype)
            #空间随机采样
            u = (torch.min(U_input)-add_border + (
                torch.max(U_input)-torch.min(U_input)+2*add_border) * (range_/number)).reshape(number,1)
            v = (torch.min(V_input)-add_border + (
                torch.max(V_input)-torch.min(V_input)+2*add_border) * (range_/number)).reshape(number,1)
            w = (torch.min(W_input)-add_border + (
                torch.max(W_input)-torch.min(W_input)+2*add_border) * (range_/number)).reshape(number,1)
            x_in = torch.cat((u,v,w),dim=1)
            out = model(centre,x_in,flag = 2).detach().cpu().clone()
            idx = (torch.where(torch.abs(out)<thres)) #选取0平面上的点:|out|=0
            Pts = torch.cat((u[idx[0]], v[idx[1]]), dim=1)
            Pts = torch.cat((Pts, w[idx[2]]), dim = 1).detach().cpu().clone().numpy()
            
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

if __name__ == '__main__':
    trainLatentCode()
    findLatentCode()
