import matplotlib.pyplot as plt
import re


def read_data_from_file(file_path):
    data = {}
    current_q = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('q='):
                current_q = float(line.split('=')[1])
                data[current_q] = {}
            elif line.startswith('SR:'):
                # 修正后的正则表达式，去除可能的逗号
                parts = re.findall(r'SR: (\S+),? PSNR: (\S+),? SSIM: (\S+),? NRMSE: (\S+)', line)[0]
                sr = float(parts[0].replace(',', ''))
                psnr = float(parts[1].replace(',', ''))
                data[current_q][sr] = psnr
    return data


def plot_data(data):
    colors = {
        0.10: '#F49E39',
        0.15: '#E7483D',
        0.20: '#918AC2',
        0.25: '#8FC751'
    }
    for sr, color in colors.items():
        q_values = []
        psnr_values = []
        for q, sub_data in data.items():
            if sr in sub_data:
                q_values.append(q)
                psnr_values.append(sub_data[sr])
        plt.plot(q_values, psnr_values, label=f'SR={sr}', color=color)

    plt.xlabel('q')
    plt.ylabel('PSNR')
    plt.title('PSNR vs q for different SR')
    plt.legend()
    plt.show()


file_path = 'log_Inpainting.txt'
data = read_data_from_file(file_path)
plot_data(data)