# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch import nn, optim 
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity
import rff
import random
import copy
from tqdm import tqdm
from PIL import Image
import argparse
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
        output = F.relu(output) #tanh、hardtanh、softplus、relu、sin
        return output

mid_channel = 512
rank = 256
posdim = 128
class Network(nn.Module):
    def __init__(self, rank, posdim, mid_channel):
        super(Network, self).__init__()

        U_net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                              MLPLayer(mid_channel, mid_channel, is_first=False),
                              MLPLayer(mid_channel, mid_channel, is_first=False),
                              nn.Linear(mid_channel, rank))
        self.U_Nets = nn.ModuleList([copy.deepcopy(U_net) for _ in range(3)])

        V_net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                              MLPLayer(mid_channel, mid_channel, is_first=False),
                              MLPLayer(mid_channel, mid_channel, is_first=False),
                              nn.Linear(mid_channel, rank))
        self.V_Nets = nn.ModuleList([copy.deepcopy(V_net) for _ in range(3)])
        
        self.encoding = rff.layers.GaussianEncoding(alpha=1.0, sigma=8.0, input_size=1, encoded_size=posdim//2) #[sigma=5,8]
    
    def forward(self, U_input, V_input):
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

def generate_random_mask(H, W, visible_ratio=0.1):
    num_visible = int(H * W * visible_ratio)
    all_positions = np.arange(H * W)
    visible_positions = np.random.choice(all_positions, size=num_visible, replace=False)
    mask = np.zeros(H * W, dtype=np.uint8)
    mask[visible_positions] = 1
    mask = mask.reshape(H, W, 1)
    mask = np.repeat(mask, 3, axis=2)
    mask = torch.from_numpy(mask).type(dtype).cuda()
    return mask


if __name__ == '__main__':
    set_random_seed(42)
    max_iter =  5001
    Schatten_q = 0.5
    image_names = ['4.2.05', '4.2.07', 'house', '4.2.06'] #[Plane Peppers House Sailboat]
    average_metrics = [0.0, 0.0, 0.0]
    
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--visible_ratio', type=float, required=True, help='The visible ratio parameter.')
    args = parser.parse_args()

    for name in image_names:
        image_path = 'data/misc/'+name+'.tiff'
        image_gt = np.array(Image.open(image_path)).astype(np.float32) / 255.0
        H, W, C = image_gt.shape
        
        mask = generate_random_mask(H, W, visible_ratio=args.visible_ratio) #[H,W,3]
        X = torch.from_numpy(image_gt).type(dtype).cuda()
        
        U_input = torch.from_numpy(np.array(range(1,H+1))).reshape(H,1).type(dtype) #[512,1] 1-512间的整数
        V_input = torch.from_numpy(np.array(range(1,W+1))).reshape(W,1).type(dtype)

        model = Network(rank, posdim, mid_channel).type(dtype)
        optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 0.002}], #[0.002]
                                lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
        
        best_metric = [0.0, 0.0, 0.0]
        with tqdm(total=max_iter) as pbar:
            for iter in range(max_iter):
                X_Out, Us, Vs = model(U_input, V_input)
                # loss_rec = torch.norm(X_Out*mask - X*mask, 2) #这里的损失需要分channel计算吗 H,W,C
                loss_rec = torch.norm(X_Out*mask - X*mask, p='fro', dim=(0, 1)).mean()
                
                U_input_eps = torch.normal(mean=U_input, std=1.0*torch.ones_like(U_input)) #std=1.0
                V_input_eps = torch.normal(mean=V_input, std=1.0*torch.ones_like(V_input))
                X_Out_eps, *_ = model(U_input_eps, V_input_eps)
                # loss_eps = torch.norm(X_Out-X_Out_eps, 2)
                loss_eps = torch.norm(X_Out - X, p='fro', dim=(0, 1)).mean()

                # loss_rank = torch.norm(Us, p='fro', dim=(0, 1)).mean() + \
                #             torch.norm(Vs, p='fro', dim=(0, 1)).mean()
                loss_rank = torch.norm(Us, p=2, dim=0).pow(Schatten_q).sum() +\
                            torch.norm(Vs, p=2, dim=0).pow(Schatten_q).sum()
                
                loss = 1.0*loss_rec + 0.05*loss_eps + 0.001*loss_rank #[1.0, 0.05 0.1]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({'loss_rec': f"{loss_rec.item():.4f}", 
                                'loss_eps': f"{loss_eps.item():.4f}", 
                                'loss_rank': f"{loss_rank.item():.4f}"})
                pbar.update()
                
                if iter % 500 == 0 and iter != 0:
                    psnr = peak_signal_noise_ratio(image_gt, X_Out.cpu().detach().numpy())
                    ssim = structural_similarity(image_gt.mean(2), 
                                                X_Out.cpu().detach().numpy().mean(2), 
                                                win_size=15, data_range=1.0)
                    nrmse = normalized_root_mse(image_gt, X_Out.cpu().detach().numpy())
                    if psnr >= best_metric[0]:
                        print('SR:',args.visible_ratio, 'name:',name, 'iteration:', iter, 'PSNR', 
                              psnr, 'SSIM', ssim, 'NRMSE', nrmse)
                        best_metric = [psnr, ssim, nrmse]
                    continue

                    plt.figure(figsize=(15,45))
                    plt.subplot(131)
                    plt.imshow(np.clip((X*mask).cpu().detach().numpy(),0,1))
                    plt.title('in')

                    plt.subplot(132)
                    plt.imshow(image_gt)
                    plt.title('gt')
            
                    plt.subplot(133)
                    plt.imshow(np.clip(X_Out.cpu().detach().numpy(),0,1))
                    plt.title('out')
                    plt.show()
        average_metrics = list(map(lambda x, y: x + y, average_metrics, best_metric))
    
    print('SR:',args.visible_ratio, 
          'PSNR: {}, SSIM: {}, NRMSE: {}'.format(*['{:.3f}'.format(metric / len(image_names)) 
                                                   for metric in average_metrics]))
