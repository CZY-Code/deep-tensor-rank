import numpy as np 
import matplotlib.pyplot as plt 
import open3d as o3d


if __name__ == '__main__':
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