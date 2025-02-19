import numpy as np 
import matplotlib.pyplot as plt 
import open3d as o3d
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image

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


def draw():
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


def zoom_fig():
    # H=64
    # W=128
    S = 4
    image_path = './temp'
    for file in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, file))
        H, W, _ = image.shape
        height = int(H/(2*S))
        width = int(W/S)
        if image is None:
            raise FileNotFoundError(f"无法找到图像文件: {image_path}")

        # 第一个局部放大图
        x1, y1 = 120, 260 # 长方形框左上角坐标
        cv2.rectangle(image, (x1, y1), (x1+width, y1+height), (0, 0, 255), 1)
        patch1 = image[y1:y1+height, x1:x1+width, :]
        patch1 = cv2.resize(patch1, (W//2, H//4)) #(256, 120)

        # 第二个局部放大图
        x2, y2 = 270, 450
        cv2.rectangle(image, (x2, y2), (x2+width, y2+height), (0, 255, 0), 1)
        patch2 = image[y2:y2+height, x2:x2+width, :]
        patch2 = cv2.resize(patch2,  (W//2, H//4)) 

        # 拼接
        patch = np.hstack((patch1, patch2))
        image = np.vstack((image, patch))

        # 显示和保存图像
        cv2.imshow('demo', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(image_path,'demo'+file), image)

def crop_center(image, crop_width, crop_height):
    """Crop the center of the image."""
    img_width, img_height = image.size
    left = (img_width - crop_width) / 2
    top = (img_height - crop_height) / 2
    right = (img_width + crop_width) / 2
    bottom = (img_height + crop_height) / 2
    return image.crop((left, top, right, bottom))

def process_images_in_place(folder_path, size=(320, 320)):
    """Process all images in a folder and save the cropped center back to the original file location."""
    folder = Path(folder_path)
    for file_path in folder.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:  # 添加更多格式如需要
            try:
                with Image.open(file_path) as img:
                    cropped_img = crop_center(img, *size)
                    # Save the cropped image back to the original file path
                    cropped_img.save(file_path)
                    print(f"Processed and saved: {file_path}")
            except IOError as e:
                print(f"Cannot create thumbnail for {file_path}: {e}")

if __name__ == '__main__':
    # plot_3d_gaussian()
    # draw()
    # zoom_fig()
    process_images_in_place('./output/Ours/Upsampling_320/')
    pass