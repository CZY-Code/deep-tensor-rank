import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch import nn, optim 
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity
import rff
import random
from tqdm import tqdm
import argparse
from PIL import Image
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
rank = 1024
posdim = 128
class Network(nn.Module):
    def __init__(self, rank, posdim, mid_channel): #
        super(Network, self).__init__()

        self.U_Net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                #    MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, rank))

        self.V_Net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True),
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                #    MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, rank))
        
        self.W_Net = nn.Sequential(MLPLayer(posdim, mid_channel, is_first=True), 
                                   MLPLayer(mid_channel, mid_channel, is_first=False),
                                #    MLPLayer(mid_channel, mid_channel, is_first=False),
                                   nn.Linear(mid_channel, rank))
        
        self.encoding = rff.layers.GaussianEncoding(alpha=1.0, sigma=8.0, input_size=1, encoded_size=posdim//2) #[sigma=5,8]

    def forward(self, U_input, V_input, W_input):
        U = self.U_Net(self.encoding(self.normalize_to_01(U_input)))
        V = self.V_Net(self.encoding(self.normalize_to_01(V_input)))
        W = self.W_Net(self.encoding(self.normalize_to_01(W_input)))
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

def read_yuv_video(yuv_filename, width, height, frame_count, color_format='420'):
    if color_format == '420':
        frame_size = (width * height) + (width // 2 * height // 2 * 2)
    with open(yuv_filename, 'rb') as f:
        raw_data = f.read()
    video_data = np.zeros((height, width, frame_count), dtype=np.float32)
    # 逐帧读取并解析 YUV 数据
    for frame_idx in range(frame_count):
        start_idx = frame_idx * frame_size
        end_idx = start_idx + frame_size
        # 读取当前帧的 YUV 数据
        frame_data = raw_data[start_idx:end_idx]
        # 提取 Y, U, V 分量
        y_data = frame_data[:width * height]
        uv_data = frame_data[width * height:]
        # 将 Y 分量转换为二维数组
        y_frame = np.frombuffer(y_data, dtype=np.uint8).reshape(height, width)
        # 将 U, V 分量转换为二维数组
        # if color_format == '420':
        #     u_data = uv_data[:width * height // 4]
        #     v_data = uv_data[width * height // 4:]
        #     u_frame = np.frombuffer(u_data, dtype=np.uint8).reshape(height // 2, width // 2)
        #     v_frame = np.frombuffer(v_data, dtype=np.uint8).reshape(height // 2, width // 2)
        #     u_frame = np.repeat(np.repeat(u_frame, 2, axis=0), 2, axis=1)
        #     v_frame = np.repeat(np.repeat(v_frame, 2, axis=0), 2, axis=1)
        # 将 Y, U, V 分量合并为 RGB 图像（可选）
        # 你可以使用 OpenCV 或其他库将 YUV 转换为 RGB 但是这里只保留 Y 分量
        video_data[:, :, frame_idx] = y_frame / 255.0
    return video_data


if __name__ == '__main__':
    set_random_seed(42)
    max_iter =  5001
    Video_names = ['foreman', 'carphone']
    average_metrics = [0.0, 0.0, 0.0]
    OB_metrics = [0.0, 0.0, 0.0]
    Schatten_q = 1.0

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--visible_ratio', type=float, required=True, help='The visible ratio parameter.')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    for name in Video_names:
        Video_path = 'data/Videos/'+name+'.yuv'
        Video_gt = read_yuv_video(Video_path, width=176, height=144, frame_count=100)
        H, W, C = Video_gt.shape
        mask = generate_random_mask_3d(H, W, C, visible_ratio=args.visible_ratio) #[H,W,3]
        X = torch.from_numpy(Video_gt).type(dtype).cuda()

        video_incomplete = (X*mask).cpu().detach().numpy()
        ob_psnr = peak_signal_noise_ratio(Video_gt, video_incomplete)
        ob_ssim = structural_similarity(Video_gt, video_incomplete, data_range=1.0, channel_axis=2)
        ob_nrmse = normalized_root_mse(Video_gt, video_incomplete)
        print(f'SR: {args.visible_ratio:.2f}', 'name:', name,
              'OB_PSNR: {:.3f}, OB_SSIM: {:.3f}, OB_NRMSE: {:.3f}'.format(ob_psnr, ob_ssim, ob_nrmse))
        OB_metrics = list(map(lambda x, y: x + y, OB_metrics, [ob_psnr, ob_ssim, ob_nrmse]))

        U_input = torch.from_numpy(np.array(range(1,H+1))).reshape(H,1).type(dtype) #[512,1] 1-512间的整数
        V_input = torch.from_numpy(np.array(range(1,W+1))).reshape(W,1).type(dtype)
        W_input = torch.from_numpy(np.array(range(1,C+1))).reshape(C,1).type(dtype)

        model = Network(rank, posdim, mid_channel).type(dtype)
        optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 0.02}], lr=0.001) #[0.02, 0.001]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
        
        best_metric = [0.0, 0.0, 0.0]
        with tqdm(total=max_iter) as pbar:
            for iter in range(max_iter):
                X_Out, U, V, W = model(U_input, V_input, W_input)
                loss_rec = torch.norm(X_Out*mask - X*mask, p='fro')
                
                U_input_eps = torch.normal(mean=U_input, std=1.0*torch.ones_like(U_input)) #std=1.0
                V_input_eps = torch.normal(mean=V_input, std=1.0*torch.ones_like(V_input))
                W_input_eps = torch.normal(mean=W_input, std=0.1*torch.ones_like(W_input)) #0.5
                X_Out_eps, *_ = model(U_input_eps, V_input_eps, W_input_eps)
                loss_eps = torch.norm(X_Out.detach()-X_Out_eps, p='fro')

                loss_rank = torch.norm(U, p='fro') + torch.norm(V, p='fro') + torch.norm(W, p='fro')
                # loss_rank = torch.norm(U, p=2, dim=0).pow(Schatten_q).sum() +\
                #             torch.norm(V, p=2, dim=0).pow(Schatten_q).sum() +\
                #             torch.norm(W, p=2, dim=0).pow(Schatten_q).sum()
                
                loss = 1.0*loss_rec + 0.05*loss_eps + 0.05*loss_rank #[1.0, 0.05 0.05]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({'loss_rec': f"{loss_rec.item():.4f}", 
                                  'loss_eps': f"{loss_eps.item():.4f}", 
                                  'loss_rank': f"{loss_rank.item():.4f}"})
                pbar.update()
                
                if iter % 500 == 0 and iter != 0:
                    psnr = peak_signal_noise_ratio(Video_gt, X_Out.cpu().detach().numpy(), data_range=1.0)
                    ssim = structural_similarity(Video_gt, X_Out.cpu().detach().numpy(), data_range=1.0, channel_axis=2)
                    nrmse = normalized_root_mse(Video_gt, X_Out.cpu().detach().numpy())
                    
                    show=0
                    if ssim >= best_metric[1]:
                        print(f'SR: {args.visible_ratio:.2f}', 'name:',name, 'iteration:', iter, 
                              'PSNR', psnr, 'SSIM', ssim, 'NRMSE', nrmse)
                        best_metric = [psnr, ssim, nrmse]
                        if args.save:
                            output_folder = os.path.join('output/Ours/inpainting/Video')
                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)
                            output_path = os.path.join(output_folder, name + f'_SR{args.visible_ratio:.2f}_psnr{psnr:.3f}_inpainting.png')
                            img = Image.fromarray((np.clip(X_Out.cpu().detach().numpy(),0,1) * 255).astype(np.uint8)[...,show])
                            img.save(output_path)
                    continue
                    show = [0]
                    plt.figure(figsize=(15,45))
                    plt.subplot(131)
                    plt.imshow(np.clip((X*mask).cpu().detach().numpy(),0,1)[...,show], cmap='gray')
                    plt.title('in')

                    plt.subplot(132)
                    plt.imshow(Video_gt[...,show], cmap='gray')
                    plt.title('gt')
            
                    plt.subplot(133)
                    plt.imshow(np.clip(X_Out.cpu().detach().numpy(),0,1)[...,show], cmap='gray')
                    plt.title('out')
                    plt.show()
        average_metrics = list(map(lambda x, y: x + y, average_metrics, best_metric))
    
    print(f'SR: {args.visible_ratio:.2f}', 
          'OB_PSNR: {}, OB_SSIM: {}, OB_NRMSE: {}'.format(*['{:.3f}'.format(metric / len(Video_names)) 
                                                   for metric in OB_metrics]))
    print(f'SR: {args.visible_ratio:.2f}', 
          'PSNR: {}, SSIM: {}, NRMSE: {}'.format(*['{:.3f}'.format(metric / len(Video_names)) 
                                                   for metric in average_metrics]))

# PSNR: 28.852, SSIM: 0.913, NRMSE: 0.065
# PSNR: 30.133, SSIM: 0.931, NRMSE: 0.056
# PSNR: 31.142, SSIM: 0.941, NRMSE: 0.050
# PSNR: 31.761, SSIM: 0.946, NRMSE: 0.046
# PSNR: 32.203, SSIM: 0.950, NRMSE: 0.044