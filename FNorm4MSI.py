import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch import nn, optim 
import torch.nn.functional as F
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity
import rff
import random
from tqdm import tqdm
from PIL import Image
import math
import argparse

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
        # output = torch.sin(self.omega_0 * output) 
        output = F.relu(output) #tanh、hardtanh、softplus、relu、sin
        return output


mid_channel = 512
rank = 1024
posdim = 128
class Network(nn.Module):
    def __init__(self, rank, posdim, mid_channel): #
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


def generate_random_mask_3d(H, W, C, visible_ratio=0.1):
    num_visible = int(H * W * C * visible_ratio)
    all_positions = np.arange(H * W * C)
    visible_positions = np.random.choice(all_positions, size=num_visible, replace=False)
    mask = np.zeros(H * W * C, dtype=np.uint8)
    mask[visible_positions] = 1
    mask = mask.reshape(H, W, C)
    mask = torch.from_numpy(mask).type(dtype).cuda()
    return mask

def generate_random_mask_2d(H, W, C, visible_ratio=0.1):
    num_visible = int(H * W * visible_ratio)
    all_positions = np.arange(H * W)
    visible_positions = np.random.choice(all_positions, size=num_visible, replace=False)
    mask = np.zeros(H * W, dtype=np.uint8)
    mask[visible_positions] = 1
    mask = mask.reshape(H, W, 1)
    mask = np.repeat(mask, C, axis=2)
    mask = torch.from_numpy(mask).type(dtype).cuda()
    return mask

def load_grayscale_images_from_directory(directory):
    png_files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    images = []
    for file in png_files:
        file_path = os.path.join(directory, file)
        image = Image.open(file_path)  # 读取为灰度图像
        image_array = np.array(image)  # 转换为 NumPy 数组
        images.append(image_array)
    if len(images) > 0:
        max_value = 2 ** 16 - 1
        stacked_images = np.stack(images, axis=-1).astype(np.float32) / max_value
        return stacked_images
    else:
        return None

#  python FNorm4MSI.py --visible_ratio 0.1
if __name__ == '__main__':
    set_random_seed(42)
    max_iter =  5001
    MSI_names = ['toys', 'flowers']
    Schatten_q = 0.3
    # MSI_names = ['toys']
    
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--visible_ratio', type=float, required=True, help='The visible ratio parameter.')
    args = parser.parse_args()

    average_metrics = [0.0, 0.0, 0.0]
    for name in MSI_names:
        MSI_path = 'data/MSIs/'+name
        
        MSI_gt = load_grayscale_images_from_directory(MSI_path)
        H, W, C = MSI_gt.shape
        mask = generate_random_mask_3d(H, W, C, visible_ratio=args.visible_ratio) #[H,W,3]
        X = torch.from_numpy(MSI_gt).type(dtype).cuda()
        
        U_input = torch.from_numpy(np.array(range(1,H+1))).reshape(H,1).type(dtype) #[512,1] 1-512间的整数
        V_input = torch.from_numpy(np.array(range(1,W+1))).reshape(W,1).type(dtype)
        W_input = torch.from_numpy(np.array(range(1,C+1))).reshape(C,1).type(dtype)

        model = Network(rank, posdim, mid_channel).type(dtype)
        optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 0.001}], #[0.001]
                                lr=0.001) #0.001
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
        
        best_metric = [0.0, 0.0, 0.0]
        with tqdm(total=max_iter) as pbar:
            for iter in range(max_iter):
                X_Out, U, V, W = model(U_input, V_input, W_input)
                loss_rec = torch.norm(X_Out*mask - X*mask, p='fro')
                
                U_input_eps = torch.normal(mean=U_input, std=0.5*torch.ones_like(U_input)) #std=1.0
                V_input_eps = torch.normal(mean=V_input, std=0.5*torch.ones_like(V_input))
                W_input_eps = torch.normal(mean=W_input, std=0.1*torch.ones_like(W_input))
                X_Out_eps, *_ = model(U_input_eps, V_input_eps, W_input_eps)
                loss_eps = torch.norm(X_Out.detach()-X_Out_eps, p='fro')

                # loss_rank = torch.norm(U, p='fro') + torch.norm(V, p='fro') + torch.norm(W, p='fro')
                loss_rank = torch.norm(U, p=2, dim=0).pow(Schatten_q).sum() +\
                            torch.norm(V, p=2, dim=0).pow(Schatten_q).sum() +\
                            torch.norm(W, p=2, dim=0).pow(Schatten_q).sum()
                
                loss = 1.0*loss_rec + 0.01*loss_eps + 0.001*loss_rank #[1.0, 0.01 0.1]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({'loss_rec': f"{loss_rec.item():.4f}", 
                                  'loss_eps': f"{loss_eps.item():.4f}", 
                                  'loss_rank': f"{loss_rank.item():.4f}"})
                pbar.update()
                
                if iter % 500 == 0 and iter != 0:
                    psnr = peak_signal_noise_ratio(MSI_gt, X_Out.cpu().detach().numpy(), data_range=1.0)
                    ssim = structural_similarity(MSI_gt, X_Out.cpu().detach().numpy(), 
                                                win_size=15, data_range=1.0, channel_axis=2)
                    nrmse = normalized_root_mse(MSI_gt, X_Out.cpu().detach().numpy())
                    # print('name:',name, 'iteration:', iter, 'PSNR', psnr, 'SSIM', ssim, 'NRMSE', nrmse)
                    
                    if ssim >= best_metric[1]:
                        print('SR:',args.visible_ratio, 'name:',name, 'iteration:', iter, 
                              'PSNR', psnr, 'SSIM', ssim, 'NRMSE', nrmse)
                        best_metric = [psnr, ssim, nrmse]
                    continue
                    show = [15,25,30]
                    plt.figure(figsize=(15,45))
                    plt.subplot(131)
                    plt.imshow(np.clip((X*mask).cpu().detach().numpy(),0,1)[...,show])
                    plt.title('in')

                    plt.subplot(132)
                    plt.imshow(MSI_gt[...,show])
                    plt.title('gt')
            
                    plt.subplot(133)
                    plt.imshow(np.clip(X_Out.cpu().detach().numpy(),0,1)[...,show])
                    plt.title('out')
                    plt.show()
        average_metrics = list(map(lambda x, y: x + y, average_metrics, best_metric))
    
    print('SR:', args.visible_ratio, 
          'PSNR: {}, SSIM: {}, NRMSE: {}'.format(*['{:.3f}'.format(metric / len(MSI_names)) 
                                                   for metric in average_metrics]))

#toys
# SR: 0.10 PSNR: 41.074, SSIM: 0.987, NRMSE: 0.031
# SR: 0.15 PSNR: 43.448, SSIM: 0.992, NRMSE: 0.024
# SR: 0.20 PSNR: 44.772, SSIM: 0.993, NRMSE: 0.021
# SR: 0.25 PSNR: 45.444, SSIM: 0.994, NRMSE: 0.019
# SR: 0.30 PSNR: 45.969, SSIM: 0.995, NRMSE: 0.018

# flowers
# SR: 0.10 PSNR: 44.582, SSIM: 0.990, NRMSE: 0.037
# SR: 0.15 PSNR: 46.753, SSIM: 0.993, NRMSE: 0.029
# SR: 0.20 PSNR: 47.817, SSIM: 0.995, NRMSE: 0.026
# SR: 0.25 PSNR: 48.762, SSIM: 0.996, NRMSE: 0.023
# SR: 0.30 PSNR: 49.607, SSIM: 0.996, NRMSE: 0.021
