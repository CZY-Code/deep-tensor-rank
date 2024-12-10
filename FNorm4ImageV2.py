#Best PSNR 28.62
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
from torch import nn, optim 
import torch.nn.functional as F
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import math
from skimage.metrics import peak_signal_noise_ratio
import rff
import random
import copy
from tqdm import tqdm


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
# 设置随机种子
seed = 42
set_random_seed(seed)

################### 
# Here are the hyperparameters. 
# w_decay = 0.1 #3
# lr_real = 0.01 #0.0001
max_iter =  5001
# down = [2,2,1]
###################


class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=1.0*np.pi): #np.pi
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                nn.init.uniform_(self.linear.weight, -1 / self.in_features, 1 / self.in_features)
            else:
                # self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                            #  np.sqrt(6 / self.in_features) / self.omega_0)
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        output = self.linear(input)
        # output = torch.sin(self.omega_0 * output) 
        output = F.relu(output) #tanh、hardtanh、softplus、relu、sin
        return output

class Network(nn.Module):
    def __init__(self, r_1,r_2,r_3, posdim):
        super(Network, self).__init__()

        U_net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                              MLPLayer(mid_channel, mid_channel, is_first=False),
                              MLPLayer(mid_channel, mid_channel, is_first=False),
                              nn.Linear(mid_channel, r_1))
        self.U_Nets = nn.ModuleList([copy.deepcopy(U_net) for _ in range(3)])

        V_net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                              MLPLayer(mid_channel, mid_channel, is_first=False),
                              MLPLayer(mid_channel, mid_channel, is_first=False),
                              nn.Linear(mid_channel, r_2))
        self.V_Nets = nn.ModuleList([copy.deepcopy(V_net) for _ in range(3)])
        
        # self.W_net = nn.Sequential(SineLayer(posdim, mid_channel, is_first=True),
        #                            SineLayer(mid_channel, mid_channel, is_first=True),
        #                            nn.Linear(mid_channel, r_3),
        #                            nn.SELU())
        
        self.encoding = rff.layers.GaussianEncoding(alpha=1.0, sigma=8.0, input_size=1, encoded_size=posdim//2) #[sigma=5]
    
    def forward(self, centre, U_input, V_input):
        # U = self.U_net(U_input)
        # V = self.V_net(V_input)
        # W = self.W_net(W_input)
        outputs = []
        outputs_U = []
        outputs_V = []
        for i in range(3):
            U = self.U_Nets[i](self.encoding(self.normalize_to_01(U_input)))
            V = self.V_Nets[i](self.encoding(self.normalize_to_01(V_input)))
            # channelImg = torch.einsum('ab,ia,jb -> ij', centre[...,i], U, V)
            channelImg = U @ V.t()
            outputs.append(channelImg)
            outputs_U.append(U)
            outputs_V.append(V)
        output = torch.stack(outputs, dim=-1)
        Us = torch.stack(outputs_U, dim=-1)
        Vs = torch.stack(outputs_V, dim=-1)
        return output, Us, Vs
    
    def normalize_to_01(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val == min_val:
            return torch.zeros_like(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor


if __name__ == '__main__':
    file_name = 'data/planep2.mat'
    mat = scipy.io.loadmat(file_name)
    X_np = mat["Nhsi"]
    X = torch.from_numpy(X_np).type(dtype).cuda()
    [n_1,n_2,n_3] = X.shape
        
    mid_channel = int(n_2)//1 #512
    r_1 = int(n_1/2)
    r_2 = int(n_2/2)
    r_3 = int(n_3/1)
    
    file_name = 'data/planegt.mat'
    mat = scipy.io.loadmat(file_name)
    gt_np = mat["Ohsi"]
    gt = torch.from_numpy(gt_np).type(dtype).cuda()
    
    mask = torch.ones(X.shape).type(dtype)
    mask[X == 0] = 0
    
    centre = torch.Tensor(r_1, r_2, r_3).type(dtype)
    stdv = 1 / math.sqrt(centre.size(0))
    centre.data.uniform_(-stdv, stdv)
    centre.requires_grad=True

    U_input = torch.from_numpy(np.array(range(1,n_1+1))).reshape(n_1,1).type(dtype) #[512,1] 为[1,512]的整数
    V_input = torch.from_numpy(np.array(range(1,n_2+1))).reshape(n_2,1).type(dtype)
    # W_input = torch.from_numpy(np.array(range(1,n_3+1))).reshape(n_3,1).type(dtype)

    model = Network(r_1, r_2, r_3, posdim=128).type(dtype)
    optimizer = optim.Adam([{'params': [centre], 'weight_decay': 0.5}, #0.5
                            {'params': model.parameters(), 'weight_decay': 0.002}], #[0.002]
                            lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
    
    with tqdm(total=max_iter) as pbar:
        for iter in range(max_iter):
            X_Out, Us, Vs = model(centre, U_input, V_input)
            # loss_rec = torch.norm(X_Out*mask - X*mask, 2) #这里的损失需要分channel计算吗 H,W,C
            loss_rec = torch.norm(X_Out*mask - X*mask, p='fro', dim=(0, 1)).mean()
            
            
            U_input_eps = torch.normal(mean=U_input, std=1.0*torch.ones_like(U_input)) #std=1.0
            V_input_eps = torch.normal(mean=V_input, std=1.0*torch.ones_like(V_input))
            X_Out_eps, *_ = model(centre, U_input_eps, V_input_eps)
            # loss_eps = torch.norm(X_Out-X_Out_eps, 2)
            loss_eps = torch.norm(X_Out - X, p='fro', dim=(0, 1)).mean()

            # loss_rank = torch.norm(Us, 2) + torch.norm(Vs, 2)
            loss_rank = torch.norm(Us, p='fro', dim=(0, 1)).mean() + \
                        torch.norm(Vs, p='fro', dim=(0, 1)).mean()
            
            loss = 1.0*loss_rec + 0.05*loss_eps + 0.1*loss_rank #[1.0, 0.05 0.1]
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss_rec': f"{loss_rec:.4f}", 'loss_eps': f"{loss_eps:.4f}", 'loss_rank': f"{loss_rank:.4f}"})
            pbar.update()
            
            if iter % 500 == 0 and iter != 0:
                ps = peak_signal_noise_ratio(np.clip(gt.cpu().detach().numpy(), 0, 1), X_Out.cpu().detach().numpy())
                print('iteration:', iter, 'PSNR', ps)
                continue

                plt.figure(figsize=(15,45))
                show = [0,1,2] 
                plt.subplot(121)
                plt.imshow(np.clip(np.stack((gt[:,:,show[0]].cpu().detach().numpy(),
                                    gt[:,:,show[1]].cpu().detach().numpy(),
                                    gt[:,:,show[2]].cpu().detach().numpy()),2),0,1))
                plt.title('gt')
        
                plt.subplot(122)
                plt.imshow(np.clip(np.stack((X_Out[:,:,show[0]].cpu().detach().numpy(),
                                    X_Out[:,:,show[1]].cpu().detach().numpy(),
                                    X_Out[:,:,show[2]].cpu().detach().numpy()),2),0,1))
                plt.title('out')
                plt.show()