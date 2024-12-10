#Best PSNR 28.69
import torch
from torch import nn, optim 
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import math
from skimage.metrics import peak_signal_noise_ratio
import rff
import random

def set_random_seed(seed):
    # 固定 PyTorch 的随机种子
    torch.manual_seed(seed)
    # 固定 CUDA 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # 固定 NumPy 的随机种子
    np.random.seed(seed)
    # 固定 Python 的随机模块的随机种子
    random.seed(seed)

# 设置随机种子
seed = 42
set_random_seed(seed)

data_all =["data/plane"]
c_all = ["2"]

################### 
# Here are the hyperparameters. 
w_decay = 2.0 #3
lr_real = 0.01 #0.0001
max_iter =  3001
down = [2,2,1]
###################


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=1.0*np.pi):  #np.pi
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        output = self.linear(input)
        output = torch.sin(self.omega_0 * output)
        return output

class Network(nn.Module):
    def __init__(self, r_1,r_2,r_3, posdim):
        super(Network, self).__init__()

        self.U_net = nn.Sequential(SineLayer(posdim, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_1),
                                   nn.SELU())
        
        self.V_net = nn.Sequential(SineLayer(posdim, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_2),
                                   nn.SELU())
        
        self.W_net = nn.Sequential(SineLayer(posdim, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_3),
                                   nn.SELU())
        
        self.encoding = rff.layers.GaussianEncoding(2.0, 1, posdim//2) #sigma不能太大，b=3
    
    def forward(self, centre, U_input, V_input, W_input):
        # U = self.U_net(U_input)
        # V = self.V_net(V_input)
        # W = self.W_net(W_input)
        U = self.U_net(self.encoding(self.normalize_to_01(U_input)))
        V = self.V_net(self.encoding(self.normalize_to_01(V_input)))
        W = self.W_net(self.encoding(self.normalize_to_01(W_input)))
        output = torch.einsum('abc,ia,jb,kc -> ijk', centre, U, V, W)
        return output
    
    def normalize_to_01(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val == min_val:
            return torch.zeros_like(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

for data in data_all:
    for c in c_all:
        file_name = data+'p'+c+'.mat'
        mat = scipy.io.loadmat(file_name)
        X_np = mat["Nhsi"]
        X = torch.from_numpy(X_np).type(dtype).cuda()
        [n_1,n_2,n_3] = X.shape
         
        mid_channel = int(n_2)//1 #512
        r_1 = int(n_1/down[0])
        r_2 = int(n_2/down[1])
        r_3 = int(n_3/down[2])
        
        file_name = data+'gt.mat'
        mat = scipy.io.loadmat(file_name)
        gt_np = mat["Ohsi"]
        gt = torch.from_numpy(gt_np).type(dtype).cuda()
        
        mask = torch.ones(X.shape).type(dtype)
        mask[X == 0] = 0
        
        centre = torch.Tensor(r_1, r_2, r_3).type(dtype)
        stdv = 1 / math.sqrt(centre.size(0))
        centre.data.uniform_(-stdv, stdv)
        U_input = torch.from_numpy(np.array(range(1,n_1+1))).reshape(n_1,1).type(dtype) #[512,1]
        V_input = torch.from_numpy(np.array(range(1,n_2+1))).reshape(n_2,1).type(dtype)
        W_input = torch.from_numpy(np.array(range(1,n_3+1))).reshape(n_3,1).type(dtype)

        model = Network(r_1, r_2, r_3, posdim=256).type(dtype)
        params = []
        params += [x for x in model.parameters()]
        centre.requires_grad=True
        params += [centre]
        optimizer = optim.Adam(params, lr=lr_real, weight_decay=w_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
        
        for iter in range(1, max_iter):
            X_Out = model(centre, U_input, V_input, W_input)
            loss = torch.norm(X_Out*mask - X*mask, 2)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
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