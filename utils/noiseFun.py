import numpy as np
import random

def add_gaussian_noise(data, std_dev):
    """Add Gaussian noise to the data."""
    gaussian_noise = np.random.normal(0, std_dev, data.shape)
    noisy_data = data + gaussian_noise
    return np.clip(noisy_data, 0, 1)  # 确保数据值在 [0, 1] 范围内

def add_sparse_noise(data, sparsity_ratio):
    """Add sparse noise to the data with a given sparsity ratio."""
    mask = np.random.rand(*data.shape) < sparsity_ratio
    sparse_noise = np.random.uniform(-1, 1, data.shape) * mask
    noisy_data = data + sparse_noise
    return np.clip(noisy_data, 0, 1)  # 确保数据值在 [0, 1] 范围内

def add_deadline_noise(data, pixel_ratio):
    """Simulate deadlines by setting some pixels to 0 across all dimensions."""
    mask = np.random.rand(*data.shape) < pixel_ratio
    data[mask] = 0
    return data

def add_stripe_noise(data, stripe_band_ratio, stripe_column_ratio):
    """Add stripe noise to a specified percentage of spectral bands and set a percentage of columns to 0."""
    num_bands = data.shape[2]
    num_stripes = int(num_bands * stripe_band_ratio)
    stripe_bands = random.sample(range(num_bands), num_stripes)

    for band in stripe_bands:
        num_columns = data.shape[1]
        num_stripe_columns = int(num_columns * stripe_column_ratio)
        stripe_columns = random.sample(range(num_columns), num_stripe_columns)
        
        for col in stripe_columns:
            data[:, col, band] = 0
    
    return data


if __name__ == '__main__':
    # 假设 clean_data 是你的原始高光谱数据
    clean_data = np.random.rand(100, 100, 200)  # 示例数据，实际数据应替换此行

    # Case 1: Gaussian noise with standard deviation 0.2
    case_1 = add_gaussian_noise(clean_data.copy(), std_dev=0.2)

    # Case 2: Gaussian noise with standard deviation 0.1 and sparse noise with SR 0.1
    case_2 = add_gaussian_noise(clean_data.copy(), std_dev=0.1)
    case_2 = add_sparse_noise(case_2, sparsity_ratio=0.1)

    # Case 3: Same noise as Case 2 plus deadlines (random pixels are set to 0)
    case_3 = case_2.copy()
    case_3 = add_deadline_noise(case_3, pixel_ratio=0.1)  # 例如，10% 的像素变为 0

    # Case 4: Same noise as Case 2 plus stripe noise in 40% of spectral bands (10% of columns are set to 0)
    case_4 = case_2.copy()
    case_4 = add_stripe_noise(case_4, stripe_band_ratio=0.4, stripe_column_ratio=0.1)

    # Case 5: Same noise as Case 3 plus stripe noise in 40% of spectral bands (10% of columns are set to 0)
    case_5 = case_3.copy()
    case_5 = add_stripe_noise(case_5, stripe_band_ratio=0.4, stripe_column_ratio=0.1)