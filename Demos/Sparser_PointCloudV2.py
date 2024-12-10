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

decoder = Decoder().cuda()
#downsampling for learning latent code#
def trainLatentCode():
    decoder.train()
    iterNum4L = 1000
    gamma_1 = 0.5
    gamma_2 = 2.5
    gamma_3 = 1.2

    params = []
    params += [x for x in decoder.parameters()]
    optimizer = optim.Adam(params, lr=lr_real)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterNum4L, eta_min=0)

    rand_num = 30 #
    add_border = 0.1

    with tqdm(total=iterNum4L) as pbar:
        for iter in range(iterNum4L):
            #空间位置上的0.5倍稀疏采样
            m = int(n*0.8)
            sampled_data = data_np[np.random.choice(n, size=m, replace=False)]
            X_axis = (torch.from_numpy(sampled_data[:,0])).reshape(m,1).type(dtype)
            # X_axis.requires_grad=True
            Y_axis = (torch.from_numpy(sampled_data[:,1])).reshape(m,1).type(dtype)
            # Y_axis.requires_grad=True
            Z_axis = (torch.from_numpy(sampled_data[:,2])).reshape(m,1).type(dtype)
            # Z_axis.requires_grad=True
            XYZaxis = torch.cat((X_axis, Y_axis, Z_axis), dim=1)

            X_Out, nuNormLoss = decoder(XYZaxis, mode='pointWise')
            loss_0 = torch.norm(X_Out - X_gt, 1) #和全零gt的L1距离
            loss_1 = gamma_1 * nuNormLoss

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
            X_Out_off, _ = decoder(axis_random, mode='Tucker')
            grad_ = gradient(X_Out_off, axis_random) #[rand_num, 3]
            loss_2 = gamma_2 * torch.norm(grad_.norm(dim=-1) - rand_num**2, 1) #norm函数默认是F范数，积分变L1范数
            loss_3 = gamma_3 * torch.norm(torch.exp(-torch.abs(X_Out_off)), 1)

            # loss = loss_0 + loss_1 + loss_2 + loss_3
            loss = loss_1 + loss_2 + loss_3
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss_0': f"{loss_0:.4f}", 'loss_1': f"{loss_1:.4f}", 'loss_2': f"{loss_2:.4f}", 'loss_3': f"{loss_3:.4f}"})
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

                out = decoder(x_in, mode='Tucker')[0].detach().cpu().clone()
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


if __name__ == '__main__':
    trainLatentCode()
