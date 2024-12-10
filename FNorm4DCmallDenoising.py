import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch import nn, optim 
import torch.nn.functional as F
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 

import rff
import random
from tqdm import tqdm
import math
import argparse
import tifffile

from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity
from utils.noiseFun import add_gaussian_noise, add_sparse_noise, add_deadline_noise, add_stripe_noise

def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()
    
    def forward(self, x, lam):
        x_abs = x.abs()-lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out

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
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        output = self.linear(input)
        output = F.relu(output) #tanh、hardtanh、softplus、relu、sin
        return output


mid_channel = 512
rank = 1024
posdim = 128
class Network(nn.Module):
    def __init__(self, rank, posdim, mid_channel):
        super(Network, self).__init__()

        self.U_Net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, rank))

        self.V_Net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, rank))
        
        self.W_Net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True), 
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, rank))
        
        self.centre = torch.Tensor(rank,rank,rank).type(dtype)
        self.centre.data.uniform_(-1 / math.sqrt(rank), 1 / math.sqrt(rank))
        
        self.encodingUV = rff.layers.GaussianEncoding(alpha=1.0, sigma=16.0, input_size=1, encoded_size=posdim//2)
        # self.encodingW = rff.layers.GaussianEncoding(alpha=1.0, sigma=32.0, input_size=1, encoded_size=posdim//4)

    def forward(self, U_input, V_input, W_input):
        U = self.U_Net(self.encodingUV(self.normalize_to_01(U_input)))
        V = self.V_Net(self.encodingUV(self.normalize_to_01(V_input)))
        W = self.W_Net(self.encodingUV(self.normalize_to_01(W_input)))
        output = torch.einsum('ir,jr,kr -> ijk', U, V, W)
        return output, U, V, W
    
    def normalize_to_01(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val == min_val:
            return torch.zeros_like(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

def truncated_linear_stretch(image, truncated_value=2):
    def gray_process(gray, maxout=1, minout=0):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray_new = (gray - truncated_down) / ((truncated_up - truncated_down) / (maxout - minout))
        gray_new[gray_new < minout] = minout
        gray_new[gray_new > maxout] = maxout
        return np.float32(gray_new)

    height, width, band = image.shape
    out =np.zeros((height, width, band))
    
    for b in range(band):
        out[:,:,b] = gray_process(image[:,:,b])
    return out

if __name__ == '__main__':
    set_random_seed(42)
    max_iter =  5001
    MSI_path = 'data/dc.tif'
    MSI_DCmall = tifffile.imread(MSI_path).transpose(1,2,0).astype(np.float32)
    MSI_DCmall = truncated_linear_stretch(MSI_DCmall)
     
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--case', type=int, default=1)
    args = parser.parse_args()

    phi = 5*10e-6 #5*10e-6
    mu = 1.0 #1.0
    gamma = 0.02 #0.02
    soft_thres = soft()

    average_metrics = [0.0, 0.0, 0.0]
    OB_metrics = [0.0, 0.0, 0.0]
    for i in range(5):
        MSI_gt = MSI_DCmall[(256*i):(256*(i+1)),:,:] #[256, 307, 191]

        if args.case == 1:
            MSI_gt_noise = add_gaussian_noise(MSI_gt.copy(), std_dev=0.2)
        elif args.case == 2:
            MSI_gt_noise = add_gaussian_noise(MSI_gt.copy(), std_dev=0.1)
            MSI_gt_noise = add_sparse_noise(MSI_gt_noise, sparsity_ratio=0.1)
        elif args.case == 3:
            MSI_gt_noise = add_gaussian_noise(MSI_gt.copy(), std_dev=0.1)
            MSI_gt_noise = add_sparse_noise(MSI_gt_noise, sparsity_ratio=0.1)
            MSI_gt_noise = add_deadline_noise(MSI_gt_noise, pixel_ratio=0.1)
        elif args.case == 4:
            MSI_gt_noise = add_gaussian_noise(MSI_gt.copy(), std_dev=0.1)
            MSI_gt_noise = add_sparse_noise(MSI_gt_noise, sparsity_ratio=0.1)
            MSI_gt_noise = add_stripe_noise(MSI_gt_noise, stripe_band_ratio=0.4, stripe_column_ratio=0.1)
        elif args.case == 5:
            MSI_gt_noise = add_gaussian_noise(MSI_gt.copy(), std_dev=0.1)
            MSI_gt_noise = add_sparse_noise(MSI_gt_noise, sparsity_ratio=0.1)
            MSI_gt_noise = add_deadline_noise(MSI_gt_noise, pixel_ratio=0.1)
            MSI_gt_noise = add_stripe_noise(MSI_gt_noise, stripe_band_ratio=0.4, stripe_column_ratio=0.1)
        else:
            raise NotImplementedError

        ob_psnr = peak_signal_noise_ratio(MSI_gt, MSI_gt_noise, data_range=1.0)
        ob_ssim = structural_similarity(MSI_gt, MSI_gt_noise, data_range=1.0, channel_axis=2)
        ob_nrmse = normalized_root_mse(MSI_gt, MSI_gt_noise)
        print('slice: ', i, 'OB_PSNR: {:.3f}, OB_SSIM: {:.3f}, OB_NRMSE: {:.3f}'.format(ob_psnr, ob_ssim, ob_nrmse))
        OB_metrics = list(map(lambda x, y: x + y, OB_metrics, [ob_psnr, ob_ssim, ob_nrmse]))

        H, W, C = MSI_gt.shape
        X = torch.from_numpy(MSI_gt_noise).type(dtype).cuda()
        mask = torch.ones(X.shape).type(dtype)
        # mask[X == 0] = 0
        
        U_input = torch.from_numpy(np.array(range(1, H+1))).reshape(H, 1).type(dtype)
        V_input = torch.from_numpy(np.array(range(1, W+1))).reshape(W, 1).type(dtype)
        W_input = torch.from_numpy(np.array(range(1, C+1))).reshape(C, 1).type(dtype)

        model = Network(rank, posdim, mid_channel).type(dtype)
        optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 0.001}], #[0.001]
                                lr=0.001) #0.001
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
        
        best_metric = [0.0, 0.0, 0.0]
        with tqdm(total=max_iter) as pbar:
            for iter in range(max_iter):
                X_Out, U, V, W = model(U_input, V_input, W_input)
                
                if iter == 0:
                    X_Out_exp = X_Out.detach()
                    D = torch.zeros([X.shape[0],X.shape[1],X.shape[2]]).type(dtype)
                    S = (X-X_Out.clone().detach()).type(dtype)
                    Vs = S.clone().detach().type(dtype)
                Vs = soft_thres(S + D / mu, gamma / mu)
                S = (2*(X - X_Out.clone().detach()) + mu * Vs - D)/(2 + mu)
                D = (D + mu * (S  - Vs)).clone().detach()
                
                loss_rec = torch.norm(X*mask-X_Out*mask-S*mask, 2)
                loss_rec = loss_rec + phi * torch.norm(X_Out[1:,:,:] - X_Out[:-1,:,:], 1)
                loss_rec = loss_rec + phi * torch.norm(X_Out[:,1:,:] - X_Out[:,:-1,:], 1)
                
                U_input_eps = torch.normal(mean=U_input, std=1.0*torch.ones_like(U_input))
                V_input_eps = torch.normal(mean=V_input, std=1.0*torch.ones_like(V_input))
                W_input_eps = torch.normal(mean=W_input, std=0.1*torch.ones_like(W_input))
                X_Out_eps, *_ = model(U_input_eps, V_input_eps, W_input_eps)
                loss_eps = torch.norm(X_Out-X_Out_eps, p='fro')

                loss_rank = torch.norm(U, p='fro') + torch.norm(V, p='fro') + torch.norm(W, p='fro')
                
                loss = 1.0*loss_rec + 0.01*loss_eps + 0.1*loss_rank #[1.0, 0.01 0.1]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({'loss_rec': f"{loss_rec:.4f}", 
                                  'loss_eps': f"{loss_eps:.4f}", 
                                  'loss_rank': f"{loss_rank:.4f}"})
                pbar.update()
                
                if iter % 500 == 0 and iter != 0:
                    psnr = peak_signal_noise_ratio(MSI_gt, X_Out.cpu().detach().numpy(), data_range=1.0)
                    ssim = structural_similarity(MSI_gt, X_Out.cpu().detach().numpy(), data_range=1.0, channel_axis=2)
                    nrmse = normalized_root_mse(MSI_gt, X_Out.cpu().detach().numpy())
                    # print('name:',name, 'iteration:', iter, 'PSNR', psnr, 'SSIM', ssim, 'NRMSE', nrmse)
                    
                    if ssim >= best_metric[1]:
                        print('slice_num:', i, 'iteration:', iter, 'PSNR', psnr, 'SSIM', ssim, 'NRMSE', nrmse)
                        best_metric = [psnr, ssim, nrmse]
                    # continue
                    show = [15, 25, 30]
                    plt.figure(figsize=(15, 45))
                    plt.subplot(131)
                    plt.imshow(np.clip(X.cpu().detach().numpy(), 0, 1)[...,show])
                    plt.title('in')

                    plt.subplot(132)
                    plt.imshow(MSI_gt[...,show])
                    plt.title('gt')
            
                    plt.subplot(133)
                    plt.imshow(np.clip(X_Out.cpu().detach().numpy(), 0, 1)[...,show])
                    plt.title('out')
                    plt.show()
        average_metrics = list(map(lambda x, y: x + y, average_metrics, best_metric))
    
    print('Case:', args.case, 'OB_PSNR: {}, OB_SSIM: {}, OB_NRMSE: {}'.format(*['{:.3f}'.format(metric / 5) 
                                                   for metric in OB_metrics]))
    print('Case:', args.case, 'PSNR: {}, SSIM: {}, NRMSE: {}'.format(*['{:.3f}'.format(metric /5) 
                                                   for metric in average_metrics]))
    
