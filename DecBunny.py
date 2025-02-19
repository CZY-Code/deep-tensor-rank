import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
import open3d as o3d
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

def draw_bar(model):
    number = 120
    u_in = torch.linspace(0,1,number).type(dtype).reshape(number,1)
    v_in = torch.linspace(0,1,number).type(dtype).reshape(number,1)
    w_in = torch.linspace(0,1,number).type(dtype).reshape(number,1)
    x_in = torch.cat((u_in, v_in, w_in), dim=1)
    # x_in = torch.normal(mean=x_in, std=0.01*torch.ones_like(x_in))
    out, U, V, W = model(x_in, flag = 3) #out [120, 120, 120] #U V W [120, 512]
    CPweights = torch.norm(U, p=2, dim=0) * torch.norm(V, p=2, dim=0) * torch.norm(W, p=2, dim=0)

    sorted_vector, sorted_indices = torch.sort(CPweights.detach().cpu().clone(), descending=True)
    sorted_vector_np = sorted_vector.numpy()
    sorted_indices_np = sorted_indices.numpy()
    return sorted_vector_np, sorted_indices_np

if __name__ == '__main__':
    thres = 0.05
    dataName = 'Bunny'
    pcd_gt = o3d.io.read_point_cloud('datasets/bunny/reconstruction/bun_zipper_res2.ply')
    X_np_gt = np.take(np.array(pcd_gt.points), indices=[0,2,1], axis=-1) #(8171, 3)
    add_border = 0.01
    N = X_np_gt.shape[0]
    # 归一化点云数据
    min_vals = X_np_gt.min() - add_border
    max_vals = X_np_gt.max() + add_border
    scaler = max_vals - min_vals
    X_np_gt = (X_np_gt - min_vals) / scaler
    X_np_gt = X_np_gt - np.mean(X_np_gt, axis=0) + np.array([0.5, 0.5, 0.5])
    X_np = X_np_gt[np.random.choice(N, size=int(N * 0.05), replace=False)]
    X_GT = torch.from_numpy(X_np_gt).type(dtype)

    model = Network(Rank=512, mid_channel=256, posdim=128).type(dtype)
    sorted_vector_np_1, sorted_indices_np_1 = draw_bar(model)
    # 将加载的权重应用到模型上
    model.load_state_dict(torch.load('model_weights.pth'))
    sorted_vector_np_2, sorted_indices_np_2 = draw_bar(model)

    # # 创建柱状图
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 14})  # 设置全局字体大小
    plt.figure(figsize=(6, 5))
    # 绘制第一个柱状图
    # bars_1 = plt.bar(range(len(sorted_vector_np_1)), sorted_vector_np_1, width=1.0, alpha=1.0, color='blue', label='Before optimization')
    # 绘制第二个柱状图
    bars_2 = plt.bar(range(len(sorted_vector_np_2)), sorted_vector_np_2, width=1.0, alpha=1.0, label='After optimization')
    # 设置图表标题和坐标轴标签
    # plt.title('Sorted Vector with Original Positions')
    plt.xlabel('Index')
    plt.ylabel('CP weight')
    plt.yticks([])
    plt.legend()
    # 显示图表
    # plt.show()
    plt.savefig('CPWeights.png')
    exit(0)

    U_max = U[:, sorted_indices[0]]
    V_max = V[:, sorted_indices[0]]
    W_max = W[:, sorted_indices[0]]
    out = torch.einsum('i,j,k -> ijk', U_max, V_max, W_max)

    idx = (torch.where(torch.abs(out)<thres))
    Pts = torch.cat((u_in[idx[0]], v_in[idx[1]], w_in[idx[2]]), dim = 1)
    CD, f_score, precision, recall = chamfer_distance_and_f_score(Pts, X_GT, threshold=thres, scaler=scaler)
    print('CD: {:.4f}, f_score: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(CD, f_score, precision, recall))
    nameSuffix = 'F{:.4f}_CD{:.4f}'.format(f_score, CD)
    # continue
    Pts_np = Pts.detach().cpu().clone().numpy()
    draw_3D(Pts_np, nameSuffix, save_folder=os.path.join('./output/Ours/dec', dataName))
