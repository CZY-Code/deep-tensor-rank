import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from torch import nn, optim 
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
import open3d as o3d
from tqdm import tqdm
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
                # self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.gain, 
                #                              np.sqrt(6 / self.in_features) / self.gain)
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))

    def forward(self, input):
        output = self.linear(input)
        output = F.leaky_relu(output, negative_slope=0.2) #tanh、hardtanh、softplus、relu、sin
        return output
        
mid_channel = 256
rank = 512
posdim = 128 #32
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

def generate_donut_point_cloud(n_points, r_torus=0.4, r_tube=0.1):
    """
    生成一个甜甜圈形状的点云。
    参数:
    n_points -- 点的数量
    r_torus -- 甜甜圈中心环的半径 (默认 2.0)
    r_tube -- 甜甜圈管的半径 (默认 0.5)
    返回:
    points -- 形状为 (n_points, 3) 的 NumPy 数组，表示点云中的点
    """
    # 随机生成角度和管上的位置
    theta = 2 * np.pi * np.random.rand(n_points)  # 中心环的角度
    phi = 2 * np.pi * np.random.rand(n_points)    # 管的角度
    tube_radius = r_tube * np.random.rand(n_points) + r_tube  # 管的半径，确保最小值为 r_tube
    # 计算点的位置
    x = (r_torus + tube_radius * np.cos(phi)) * np.cos(theta)
    y = (r_torus + tube_radius * np.cos(phi)) * np.sin(theta)
    z = tube_radius * np.sin(phi)
    points = np.vstack((x, y, z)).T
    return points

def generate_sphere_point_cloud(n_points, radius=0.4):
    """
    生成一个球形点云。
    参数:
    n_points -- 点的数量
    radius -- 球的半径 (默认 1.0)
    返回:
    points -- 形状为 (n_points, 3) 的 NumPy 数组，表示点云中的点
    """
    # 使用随机数生成球坐标系下的点
    phi = np.random.uniform(0, np.pi, n_points)  # 极角，范围 [0, pi]
    theta = np.random.uniform(0, 2 * np.pi, n_points)  # 方位角，范围 [0, 2*pi]
    # 将球坐标转换为笛卡尔坐标
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    points = np.vstack((x, y, z)).T
    return points

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
    save_path = os.path.join(save_folder, f'{dataName}_{nameSuffix}.png')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(save_path)
    # plt.show()

if __name__ == '__main__':
    #################
    # Here are the hyperparameters.
    thres = 0.05
    max_iter = 5000 #2001
    dataName = 'Bunny'
    #################
    if dataName == 'Bunny':
        pcd_gt = o3d.io.read_point_cloud('datasets/bunny/reconstruction/bun_zipper_res2.ply')
        X_np_gt = np.take(np.array(pcd_gt.points), indices=[0,2,1], axis=-1) #(8171, 3)
        add_border = 0.01
    elif dataName == 'Doughnut':
        X_np_gt = generate_donut_point_cloud(10000)
        add_border = 0.1
    elif dataName == 'Sphere':
        X_np_gt = generate_sphere_point_cloud(10000) #add_border = 0.1 如果点云占满整个空间，这个值太小会导致左右连接起来，因为位置编码到一个周期了
        add_border = 0.1
    elif dataName == 'Heart':
        pcd_gt = o3d.io.read_point_cloud('data/heart.pcd')
        X_np_gt = np.array(pcd_gt.points)[:,:]
        add_border = 0.1
    else:
        raise NotImplementedError

    N = X_np_gt.shape[0]
    # 归一化点云数据
    min_vals = X_np_gt.min() - add_border
    max_vals = X_np_gt.max() + add_border
    scaler = max_vals - min_vals
    X_np_gt = (X_np_gt - min_vals) / scaler
    X_np_gt = X_np_gt - np.mean(X_np_gt, axis=0) + np.array([0.5, 0.5, 0.5])
    X_np = X_np_gt[np.random.choice(N, size=int(N * 0.05), replace=False)]
    X_GT = torch.from_numpy(X_np_gt).type(dtype)

    # X_OB = torch.from_numpy(X_np).type(dtype)
    # CD, f_score, precision, recall = chamfer_distance_and_f_score(X_OB, X_GT, threshold=0.001, scaler=1)
    # print(dataName, 'CD: {:.4f}, f_score: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(CD, f_score, precision, recall))
    # exit(0)
    # draw_3D(X_np, 'ob', save_folder='./output/Origin/Upsampling')
    # draw_3D(X_np_gt, 'gt', save_folder='./output/Origin/Upsampling')
    # exit(0)
    
    n = X_np.shape[0]
    X_gt = torch.zeros(n, 1).type(dtype)
    U_input = (torch.from_numpy(X_np[:,0])).reshape(n,1).type(dtype)
    V_input = (torch.from_numpy(X_np[:,1])).reshape(n,1).type(dtype)
    W_input = (torch.from_numpy(X_np[:,2])).reshape(n,1).type(dtype)
    x_input = torch.cat((U_input, V_input, W_input), dim=1)

    model = Network(rank, mid_channel, posdim).type(dtype)
    optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 0.001}], lr=0.0001) #0.0001 同时0.001的视觉效果不错
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
    
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
            loss_2 = 0.5 * torch.norm(grad_.norm(dim=-1) - rand_num**2, 1) #梯度大小固定为1的损失 #1
            loss_3 = 6.0 * torch.norm(torch.exp(-torch.abs(X_Out_off)), 1) #其他点原理0平面的损失 #4
            loss_rec = loss_1 + loss_2 + loss_3

            #=======eps loss==================
            x_input_eps = torch.normal(mean=x_random.detach(), std=0.01*torch.ones_like(x_random))
            X_Out_eps, Us, Vs, Ws = model(x_input_eps, flag = 3) #[299,1,1]
            loss_eps = torch.norm(X_Out_eps - X_Out_off.detach(), 2)
            #=========low rank loss============
            loss_rank = torch.norm(Us, 2) + torch.norm(Vs, 2) + torch.norm(Ws, 2)

            loss = 1.0*loss_rec + 1.0*loss_eps + 1.0*loss_rank #[1.0, 0.1, 1.0] [1, 2000, 100]
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss_1': f"{loss_1:.4f}", 'loss_2': f"{loss_2:.4f}", 'loss_3': f"{loss_3:.4f}",
                                'loss_e': f"{loss_eps:.4f}", 'loss_l': f"{loss_rank:.4f}"})
            pbar.update()
            # continue
            if iter % 1000 == 0 and iter != 0:
                print('iteration:', iter)
                number = 120 #90
                u = torch.linspace(0,1,number).type(dtype).reshape(number,1)
                v = torch.linspace(0,1,number).type(dtype).reshape(number,1)
                w = torch.linspace(0,1,number).type(dtype).reshape(number,1)
                x_in = torch.cat((u, v, w), dim=1)
                # x_in = torch.normal(mean=x_in, std=0.01*torch.ones_like(x_in))
                out = model(x_in, flag = 2).detach().cpu().clone()
                idx = (torch.where(torch.abs(out)<thres))
                Pts = torch.cat((u[idx[0]], v[idx[1]]), dim = 1)
                Pts = torch.cat((Pts, w[idx[2]]), dim = 1)
                CD, f_score, precision, recall = chamfer_distance_and_f_score(Pts, X_GT, threshold=thres, scaler=scaler)
                print('CD: {:.4f}, f_score: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(CD, f_score, precision, recall))
                nameSuffix = 'F{:.4f}_CD{:.4f}'.format(f_score, CD)
                # continue
                Pts_np = Pts.detach().cpu().clone().numpy()
                draw_3D(Pts_np, nameSuffix, save_folder=os.path.join('./output/Ours/Upsampling', dataName))
                # if CD <= 0.0016 and f_score >= 0.89:
                #     save_path = 'model_weights.pth'
                #     torch.save(model.state_dict(), save_path)
                #     print(f"模型权重已保存到 {save_path}")
