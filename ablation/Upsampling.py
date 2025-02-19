import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from torch import nn, optim 
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
import open3d as o3d
from tqdm import tqdm
import argparse
import sys
sys.path.append('./')
import rff
import random
from utils.metrics import chamfer_distance_and_f_score
dtype = torch.cuda.FloatTensor


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed = 42
set_random_seed(seed)


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, gain=1.0*np.pi):
        super().__init__()
        self.gain = gain
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                nn.init.uniform_(self.linear.weight, -1 / self.in_features, 1 / self.in_features)
            else:
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))

    def forward(self, input):
        output = self.linear(input)
        output = F.leaky_relu(output, negative_slope=0.2) #tanh、hardtanh、softplus、relu、sin
        return output
        

class Network(nn.Module):
    def __init__(self, Rank, mid_channel, posdim):
        super(Network, self).__init__()
        self.posdim = posdim
        self.U_net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, Rank))
        
        self.V_net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, Rank))
        
        self.W_net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, Rank))
        
        self.encoding = rff.layers.GaussianEncoding(alpha=1.0, sigma=8.0, input_size=1, encoded_size=posdim//2) #8

    def forward(self, x, flag):
        coords = torch.stack([self.encoding(x[:,i].unsqueeze(1)) for i in range(3)], dim=-1)
        # coords = torch.stack([x[:,i].unsqueeze(1) for i in range(3)], dim=-1)
        U = self.U_net(coords[..., 0]) #[299, proR]
        V = self.V_net(coords[..., 1]) #[299, proR]
        W = self.W_net(coords[..., 2]) #[299, proR]
        if flag == 1:
            output = torch.einsum('nr,nr,nr -> n', U, V, W)
            return output
        elif flag == 2:
            output = torch.einsum('ir,jr,kr -> ijk', U, V, W)
            return output
        elif flag == 3:
            output = torch.einsum('ir,jr,kr -> ijk', U, V, W)
            return output, U, V, W
        else:
            raise NotImplementedError


if __name__ == '__main__':
    #################
    # Here are the hyperparameters.
    thres = 0.05 #0.05
    max_iter = 5000 #2001
    Schatten_q = 0.01
    ranks = [128, 256, 512, 1024]
    mid_channel = 256
    posdim = 32
    
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--name', type=str, required=True, help='The name parameter.')
    args = parser.parse_args()
    dataName = args.name
    
    rootPath = 'datasets/shapeNet/shapenetcore_partanno_segmentation_benchmark_v0'
    #################
    dataset_dict = {'Table': '04379243/points/fcd4d0e1777f4841dcfcef693e7ec696.pts',
                    'Airplane': '02691156/points/1c27d282735f81211063b9885ddcbb1.pts',
                    'Chair': '03001627/points/fd05e5d8fd82508e6d0a0d492005859c.pts',
                    'Lamp': '03636649/points/1a9c1cbf1ca9ca24274623f5a5d0bcdc.pts'}
    file_name = os.path.join(rootPath, dataset_dict[dataName])

    X_np_gt = np.take(np.loadtxt(file_name), indices=[0,2,1], axis=-1)
    N = X_np_gt.shape[0]
    print('The total number of points is: ', N)
    X_np = X_np_gt[np.random.choice(N, size=int(N * 0.2), replace=False)]
    add_border = 0.01
    
    # 归一化点云数据
    min_vals = X_np_gt.min() - add_border
    max_vals = X_np_gt.max() + add_border
    scaler = max_vals - min_vals
    X_np = (X_np - min_vals) / scaler
    X_np_gt = (X_np_gt - min_vals) / scaler
    X_GT = torch.from_numpy(X_np_gt).type(dtype)

    n = X_np.shape[0]
    X_gt = torch.zeros(n, 1).type(dtype)
    U_input = (torch.from_numpy(X_np[:,0])).reshape(n,1).type(dtype)
    V_input = (torch.from_numpy(X_np[:,1])).reshape(n,1).type(dtype)
    W_input = (torch.from_numpy(X_np[:,2])).reshape(n,1).type(dtype)
    x_input = torch.cat((U_input, V_input, W_input), dim=1)

    for rank in ranks:
        model = Network(rank, mid_channel, posdim).type(dtype)
        optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 0.001}], lr=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
        
        best_metric = [0.0, 0.0]
        rand_num = 30
        with tqdm(total=max_iter) as pbar:
            for iter in range(1, max_iter+1):
                #在[0,1]之间均匀随机取值
                U_random = torch.rand(rand_num,1).type(dtype).reshape(rand_num,1)
                U_random.requires_grad=True
                V_random = torch.rand(rand_num,1).type(dtype).reshape(rand_num,1)
                V_random.requires_grad=True
                W_random = torch.rand(rand_num,1).type(dtype).reshape(rand_num,1)
                W_random.requires_grad=True
                x_random = torch.cat((U_random,V_random, W_random), dim=1)

                X_Out = model(x_input, flag = 1) #[299,1,1]
                loss_1 = torch.norm(X_Out - X_gt, 1)
                X_Out_off = model(x_random, flag = 2) #只用于计算SDF损失 [30,30,30]
                grad_ = gradient(X_Out_off, x_random) #shape [30, 3]
                loss_2 = 1.0 * torch.norm(grad_.norm(dim=-1) - rand_num**2, 1) #梯度大小固定为1 #2 1
                loss_3 = 4.0 * torch.norm(torch.exp(-torch.abs(X_Out_off)), 1) #其他点远离零平面 #8 4
                loss_rec = loss_1 + loss_2 + loss_3

                #=======eps loss==================
                x_input_eps = torch.normal(mean=x_random.detach(), std=0.01*torch.ones_like(x_random))
                X_Out_eps, Us, Vs, Ws = model(x_input_eps, flag = 3) #[299,1,1]
                loss_eps = torch.norm(X_Out_eps - X_Out_off.detach(), 2)
                #=========low rank loss============
                # loss_rank = torch.norm(Us, 2) + torch.norm(Vs, 2) + torch.norm(Ws, 2)
                if Schatten_q == 0.0:
                    loss_rank = torch.zeros(1).type(dtype)
                else:
                    loss_rank = torch.norm(Us, p=2, dim=0).pow(Schatten_q).sum() +\
                                torch.norm(Vs, p=2, dim=0).pow(Schatten_q).sum() +\
                                torch.norm(Ws, p=2, dim=0).pow(Schatten_q).sum()

                loss = 1.0*loss_rec + 1000.0*loss_eps + 200.0*loss_rank #[1.0, 1000.0, 10.0]
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({'loss_1': f"{loss_1.item():.4f}", 'loss_2': f"{loss_2.item():.4f}", 'loss_3': f"{loss_3.item():.4f}",
                                'loss_e': f"{loss_eps.item():.4f}", 'loss_r': f"{loss_rank.item():.4f}"})
                pbar.update()
                # continue
                if iter % 1000 == 0 and iter != 0:
                    print('iteration:', iter)
                    number = 60 #90
                    u = torch.linspace(0,1,number).type(dtype).reshape(number,1)
                    v = torch.linspace(0,1,number).type(dtype).reshape(number,1)
                    w = torch.linspace(0,1,number).type(dtype).reshape(number,1)
                    x_in = torch.cat((u, v, w), dim=1)
                    out = model(x_in, flag = 2).detach().cpu().clone()
                    idx = (torch.where(torch.abs(out)<thres))
                    Pts = torch.cat((u[idx[0]], v[idx[1]]), dim = 1)
                    Pts = torch.cat((Pts, w[idx[2]]), dim = 1)
                    CD, f_score, precision, recall = chamfer_distance_and_f_score(Pts, X_GT, threshold=thres, scaler=scaler)
                    print('Name: {}, CD: {:.4f}, f_score: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(dataName, CD, f_score, precision, recall))
            
                    if f_score >= best_metric[1]:
                        best_metric = [CD, f_score]

        with open('ablation/r_Upsampling_records.txt', 'a') as file:
            file.write(f"Name: {dataName}, rank: {rank:d}, CD: {best_metric[0]:.4f}, F1: {best_metric[1]:.4f}\n")

    
