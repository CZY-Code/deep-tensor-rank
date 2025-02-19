import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def q4PSNRInpainting(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 解析数据
    data = []
    for line in lines:
        parts = line.strip().split(', ')
        sr = float(parts[0].split(': ')[1])
        schatten_q = float(parts[1].split(': ')[1])
        psnr = float(parts[2].split(': ')[1])
        data.append([sr, schatten_q, psnr])
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=['SR', 'Schatten_q', 'PSNR'])
    # 获取不同的键值
    unique_sr = np.sort(df['SR'].unique())
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 14})  # 设置全局字体大小
    plt.figure(figsize=(6, 5))
    # 绘制不同 SR 的曲线
    for sr in unique_sr:
        subset = df[df['SR'] == sr]
        x = subset['Schatten_q'].to_numpy()/3
        y = subset['PSNR'].to_numpy()
        plt.plot(x, y, label=f'SR {sr:.2f}')
    # 设置图表标题和坐标轴标签
    # plt.title('Relationship between q and PSNR in inpainting')
    plt.xlabel('p value')
    plt.ylabel('PSNR')
    # 显示图例
    plt.legend(loc=1)
    # 显示网格
    plt.grid(True)    
    plt.savefig(file_path.split('.')[0] + '.png')

def r4PSNRInpainting(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 解析数据
    data = []
    for line in lines:
        parts = line.strip().split(', ')
        sr = float(parts[0].split(': ')[1])
        schatten_q = float(parts[1].split(': ')[1])
        psnr = float(parts[2].split(': ')[1])
        data.append([sr, schatten_q, psnr])
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=['SR', 'rank', 'PSNR'])
    # 获取不同的键值
    unique_sr = np.sort(df['SR'].unique())
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 14})  # 设置全局字体大小
    plt.figure(figsize=(6, 5))
    # 绘制不同 SR 的曲线
    for sr in unique_sr:
        subset = df[df['SR'] == sr]
        x = subset['rank'].to_numpy()
        y = subset['PSNR'].to_numpy()
        plt.plot(x, y, label=f'SR {sr:.2f}')
    # 设置图表标题和坐标轴标签
    # plt.title('Relationship between q and PSNR in inpainting')
    plt.xlabel('rank R')
    plt.ylabel('PSNR')
    # 显示图例
    plt.legend(loc=4)
    # 显示网格
    plt.grid(True)    
    plt.savefig(file_path.split('.')[0] + '.png')

def q4PSNRDenoising(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 解析数据
    data = []
    for line in lines:
        parts = line.strip().split(', ')
        case = int(parts[0].split(': ')[1])
        schatten_q = float(parts[1].split(': ')[1])
        psnr = float(parts[2].split(': ')[1])
        data.append([case, schatten_q, psnr])
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=['Case', 'Schatten_q', 'PSNR'])
    # 获取不同的键值
    unique_case = np.sort(df['Case'].unique())
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 14})  # 设置全局字体大小
    plt.figure(figsize=(6, 5))
    # 绘制不同 SR 的曲线
    for case in unique_case:
        subset = df[df['Case'] == case]
        x = subset['Schatten_q'].to_numpy()/3
        y = subset['PSNR'].to_numpy()
        plt.plot(x, y, label=f'Case {case}')
    # 设置图表标题和坐标轴标签
    # plt.title('Relationship between q and PSNR in denoising')
    plt.xlabel('p value')
    plt.ylabel('PSNR')
    # 显示图例
    plt.legend(loc=1)
    # 显示网格
    plt.grid(True)    
    plt.savefig(file_path.split('.')[0] + '.png')

def r4PSNRDenoising(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 解析数据
    data = []
    for line in lines:
        parts = line.strip().split(', ')
        case = int(parts[0].split(': ')[1])
        schatten_q = float(parts[1].split(': ')[1])
        psnr = float(parts[2].split(': ')[1])
        data.append([case, schatten_q, psnr])
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=['Case', 'rank', 'PSNR'])
    # 获取不同的键值
    unique_case = np.sort(df['Case'].unique())
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 14})  # 设置全局字体大小
    plt.figure(figsize=(6, 5))
    # 绘制不同 SR 的曲线
    for case in unique_case:
        subset = df[df['Case'] == case]
        x = subset['rank'].to_numpy()
        y = subset['PSNR'].to_numpy()
        plt.plot(x, y, label=f'Case {case}')
    # 设置图表标题和坐标轴标签
    # plt.title('Relationship between q and PSNR in denoising')
    plt.xlabel('rank R')
    plt.ylabel('PSNR')
    # 显示图例
    plt.legend(loc=4)
    # 显示网格
    plt.grid(True)    
    plt.savefig(file_path.split('.')[0] + '.png')

def q4F1Upsampling(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 解析数据
    data = []
    for line in lines:
        parts = line.strip().split(', ')
        name = parts[0].split(': ')[1]
        schatten_q = float(parts[1].split(': ')[1])
        F1 = float(parts[3].split(': ')[1])
        data.append([name, schatten_q, F1])
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=['Name', 'Schatten_q', 'F1'])
    # 获取不同的键值
    unique_names = df['Name'].unique()
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 14})  # 设置全局字体大小
    plt.figure(figsize=(6, 5))
    # 绘制不同 SR 的曲线
    for name in unique_names:
        subset = df[df['Name'] == name]
        x = subset['Schatten_q'].to_numpy()/3
        y = subset['F1'].to_numpy()
        plt.plot(x, y, label=name)
    # 设置图表标题和坐标轴标签
    # plt.title('Relationship between q and F1 in upsampling')
    plt.xlabel('p value')
    plt.ylabel('F1')
    # 显示图例
    plt.legend(loc=1)
    # 显示网格
    plt.grid(True)    
    plt.savefig(file_path.split('.')[0] + '.png')

def r4F1Upsampling(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 解析数据
    data = []
    for line in lines:
        parts = line.strip().split(', ')
        name = parts[0].split(': ')[1]
        schatten_q = float(parts[1].split(': ')[1])
        F1 = float(parts[3].split(': ')[1])
        data.append([name, schatten_q, F1])
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=['Name', 'rank', 'F1'])
    # 获取不同的键值
    unique_names = df['Name'].unique()
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 14})  # 设置全局字体大小
    plt.figure(figsize=(6, 5))
    # 绘制不同 SR 的曲线
    for name in unique_names:
        subset = df[df['Name'] == name]
        x = subset['rank'].to_numpy()
        y = subset['F1'].to_numpy()
        plt.plot(x, y, label=name)
    # 设置图表标题和坐标轴标签
    # plt.title('Relationship between q and F1 in upsampling')
    plt.xlabel('rank R')
    plt.ylabel('F1')
    # 显示图例
    plt.legend(loc=4)
    # 显示网格
    plt.grid(True)    
    plt.savefig(file_path.split('.')[0] + '.png')

if __name__ == '__main__':
    # for task in ['Inpainting', 'Denoising', 'Upsampling']:
    for task in ['Denoising']:
        if task == 'Inpainting':
            # q4PSNRInpainting(file_path = 'q_Inpainting_records.txt')
            r4PSNRInpainting(file_path = 'r_Inpainting_records.txt')
        elif task == 'Denoising':
            # q4PSNRDenoising(file_path = 'Denoising_records.txt')
            r4PSNRDenoising(file_path = 'r_Denoising_records.txt')
        elif task == 'Upsampling':
            # q4F1Upsampling(file_path = 'Upsampling_records.txt')
            r4F1Upsampling(file_path = 'r_Upsampling_records.txt')
        else:
            raise ValueError('Invalid task')



    