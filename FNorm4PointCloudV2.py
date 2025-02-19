# For Heart CD: 0.0459, f_score: 0.9984, precision: 0.9979, recall: 0.9990
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch import nn, optim 
import torch.nn.functional as F
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import open3d as o3d
from tqdm import tqdm
import rff
import random
from utils.metrics import chamfer_distance_and_f_score


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
                # self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.gain, 
                #                              np.sqrt(6 / self.in_features) / self.gain)
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))

    def forward(self, input):
        output = self.linear(input)
        # output = torch.sin(self.gain * output) 
        output = F.leaky_relu(output, negative_slope=0.2) #tanh、hardtanh、softplus、relu、sin
        return output
        

class Network(nn.Module):
    def __init__(self, Rank, mid_channel, posdim):
        super(Network, self).__init__()
        self.posdim = posdim
        self.U_net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, Rank))
        
        self.V_net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, Rank))
        
        self.W_net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, Rank))
        
        self.encoding = rff.layers.GaussianEncoding(1.0, 0.5, 1, posdim//2) #sigma越小越好

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

def draw_3D(points, nameSuffix, save_folder):    
    size_pc = 6
    cmap = plt.cm.get_cmap('magma')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
    ax.scatter(xs, ys, zs,s=size_pc,c=zs, cmap=cmap)
    ax.view_init(elev=30, azim=90)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax._axis3don = False
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    # 计算中心点
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    # 设置坐标轴范围，确保等比例缩放
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])
    # 生成保存路径
    save_path = os.path.join(save_folder, f'Heart_{nameSuffix}.png')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(save_path)
    # plt.show()


if __name__ == '__main__':
    #################
    # Here are the hyperparameters.
    lr_real = 0.00001 #0.000001 
    thres = 0.01
    max_iter = 2001 #2001
    gamma_1 = 0.3
    gamma_2 = 0.3
    
    #################
    pcd = o3d.io.read_point_cloud('data/heartp0.05.pcd')
    pcd_gt = o3d.io.read_point_cloud('data/heart.pcd')
    X_np = np.array(pcd.points)[:,:]
    X_np_gt = np.array(pcd_gt.points)[:,:]
    add_border = 0.1
    
    # 归一化点云数据
    min_vals = X_np.min() - add_border
    max_vals = X_np.max() + add_border
    scaler = max_vals - min_vals
    X_np = (X_np - min_vals) / scaler
    X_np_gt = (X_np_gt - min_vals) / scaler
    X_GT = torch.from_numpy(X_np_gt).type(dtype)
    
    n = X_np.shape[0]
    mid_channel = 256 #n
    Rank = 32
    
    X_gt = torch.zeros(n,1).type(dtype)
    U_input = (torch.from_numpy(X_np[:,0])).reshape(n,1).type(dtype)
    V_input = (torch.from_numpy(X_np[:,1])).reshape(n,1).type(dtype)
    W_input = (torch.from_numpy(X_np[:,2])).reshape(n,1).type(dtype)
    x_input = torch.cat((U_input, V_input, W_input), dim=1)

    model = Network(Rank, mid_channel, posdim=64).type(dtype)
    optimizier = optim.Adam([{'params': model.parameters(), 'weight_decay': 0.0}], lr=lr_real) 
    rand_num = 30
    
    with tqdm(total=max_iter) as pbar:
        for iter in range(max_iter):
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
            loss_2 = 0.3 * torch.norm(grad_.norm(dim=-1) - rand_num**2, 1) #0.3
            loss_3 = 0.3 * torch.norm(torch.exp(-torch.abs(X_Out_off)), 1) #0.3
            loss_rec = loss_1 + loss_2 + loss_3

            #=======eps loss==================
            x_input_eps = torch.normal(mean=x_random.detach(), std=0.01*torch.ones_like(x_random)) #[0.1]
            X_Out_eps, Us, Vs, Ws = model(x_input_eps, flag = 3) #[299,1,1]
            loss_eps = torch.norm(X_Out_eps - X_Out_off, 2)
            #=========low rank loss============
            loss_rank = torch.norm(Us, 2) + torch.norm(Vs, 2) + torch.norm(Ws, 2)

            loss = 1.0*loss_rec + 1.0*loss_eps + 1.0*loss_rank #[1.0, 0.1, 1.0]
            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()
            
            pbar.set_postfix({'loss_1': f"{loss_1:.4f}", 'loss_2': f"{loss_2:.4f}", 'loss_3': f"{loss_3:.4f}",
                                'loss_e': f"{loss_eps:.4f}", 'loss_l': f"{loss_rank:.4f}"})
            pbar.update()
            # continue
            if iter % 1000 == 0 and iter != 0:
                print('iteration:', iter)
                number = 90
                u = torch.linspace(0,1,number).type(dtype).reshape(number,1)
                v = torch.linspace(0,1,number).type(dtype).reshape(number,1)
                w = torch.linspace(0,1,number).type(dtype).reshape(number,1)
                x_in = torch.cat((u, v, w),dim=1)
                out = model(x_in, flag = 2).detach().cpu().clone()
                idx = (torch.where(torch.abs(out)<thres))
                Pts = torch.cat((u[idx[0]], v[idx[1]]), dim = 1)
                Pts = torch.cat((Pts, w[idx[2]]), dim = 1)
                CD, f_score, precision, recall = chamfer_distance_and_f_score(Pts, X_GT, threshold=thres, scaler=scaler)
                print('CD: {:.4f}, f_score: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(CD, f_score, precision, recall))
                nameSuffix = 'F{:.4f}_CD{:.4f}'.format(f_score, CD)
                Pts_np = Pts.detach().cpu().clone().numpy()
                draw_3D(Pts_np, nameSuffix, save_folder=os.path.join('./output/Ours/Upsampling', 'Heart'))
                continue
                size_pc = 6
                fig = plt.figure(figsize=(15,30))
                ax = plt.subplot(121, projection='3d')
                xs = Pts_np[:,0]
                ys = Pts_np[:,1]
                zs = Pts_np[:,2]
                ax.scatter(xs, ys, zs,s=size_pc)
                ax.view_init(elev=30, azim=90)
                
                ax = fig.add_subplot(122, projection='3d')
                xs = X_np[:,0]
                ys = X_np[:,1]
                zs = X_np[:,2]
                ax.scatter(xs, ys, zs,s=size_pc)
                ax.view_init(elev=30, azim=90)
                plt.show()