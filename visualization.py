import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from matplotlib import cm

def draw_bunny():
    pcd_gt = o3d.io.read_point_cloud('datasets/bunny/reconstruction/bun_zipper_res2.ply')
    X_np_gt = np.take(np.array(pcd_gt.points), indices=[0,2,1], axis=-1) #(8171, 3)
    size_pc = 6
    fig = plt.figure(figsize=(15,15))
    cmap = plt.cm.get_cmap('magma')
    ax = plt.subplot(111, projection='3d')
    xs = X_np_gt[:,0]
    ys = X_np_gt[:,1]
    zs = X_np_gt[:,2]
    ax.scatter(xs, ys, zs,s=size_pc,c=zs, cmap=cmap)
    # 去除背景网格
    ax.grid(False)
    # 去除坐标轴平面的填充色
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # 去除坐标轴平面的边框
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # 完全移除坐标轴
    ax._axis3don = False
    ax.view_init(elev=30, azim=90)
    plt.show()

def plot_3d_gaussian():
    def gaussian_3d(x, y, mx=0, my=0, sx=1, sy=1):
        """
        生成二维高斯函数值。
        参数:
        x, y -- 坐标网格点
        mx, my -- 分布的均值 (默认为 0)
        sx, sy -- 分布的标准差 (默认为 1)
        返回:
        z -- 二维高斯函数值
        """
        return np.exp(-((x - mx)**2 / (2 * sx**2) + (y - my)**2 / (2 * sy**2)))
    # 定义坐标范围和分辨率
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)
    # 计算高斯函数值
    z = gaussian_3d(x, y)
    # 创建 3D 图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制曲面图
    surf = ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=0, antialiased=False)
    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax._axis3don = False
    # 设置标题和标签
    # ax.set_title('3D Gaussian Distribution')
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # 显示图形
    plt.show()


def test():
    def f_rugged(x, y):
        return np.sin(x) * np.cos(y) + np.random.normal(0, 0.2, size=x.shape)
    def f_smooth(x, y):
        return np.sin(x) * np.cos(y)

    # 生成空间坐标点
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    # 计算崎岖表面的值
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = f_rugged(X[i, j], Y[i, j])

    # 绘制崎岖表面
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('Rugged Surface')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('f(x,y)')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # 去除背景网格
    ax.grid(False)

    # 计算光滑表面的值
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = f_smooth(X[i, j], Y[i, j])

    # 绘制光滑表面
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('Smooth Surface')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('f(x,y)')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # 去除背景网格
    ax.grid(False)

    plt.show()

if __name__ == '__main__':
    # plot_3d_gaussian()
    test()