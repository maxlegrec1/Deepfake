import numpy as np
import torch
import torch.nn.functional as F


def dssim(img1, img2, max_val=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if img1.dtype != img2.dtype:
        raise ValueError("img1.dtype != img2.dtype")

    # Move inputs to the selected device
    img1 = img1.to(device)
    img2 = img2.to(device)

    not_float32 = img1.dtype != torch.float32

    if not_float32:
        img_dtype = img1.dtype
        img1 = img1.to(torch.float32)
        img2 = img2.to(torch.float32)

    filter_size = max(1, filter_size)

    # Create Gaussian kernel
    kernel = torch.arange(0, filter_size, dtype=torch.float32, device=device)
    kernel -= (filter_size - 1) / 2.0
    kernel = kernel**2
    kernel *= -0.5 / (filter_sigma**2)
    kernel = kernel.unsqueeze(1) + kernel.unsqueeze(0)
    kernel = torch.reshape(kernel, (1, -1))
    kernel = F.softmax(kernel, dim=1)
    kernel = kernel.reshape(1, 1, filter_size, filter_size)
    kernel = kernel.repeat(img1.shape[1], 1, 1, 1)

    def reducer(x):
        return F.conv2d(x, kernel, padding="valid", groups=x.shape[1])

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mean0 = reducer(img1)
    mean1 = reducer(img2)
    num0 = mean0 * mean1 * 2.0
    den0 = torch.square(mean0) + torch.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(img1 * img2) * 2.0
    den1 = reducer(torch.square(img1) + torch.square(img2))
    c2 *= 1.0  # compensation factor
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    ssim_val = torch.mean(luminance * cs, dim=[2, 3])
    dssim = (1.0 - ssim_val) / 2.0

    if not_float32:
        dssim = dssim.to(img_dtype)

    return dssim


def gaussian_blur(input, radius=2.0):
    def gaussian(x, mu, sigma):
        return np.exp(-((float(x) - float(mu)) ** 2) / (2 * sigma**2))

    def make_kernel(sigma):
        kernel_size = max(3, int(2 * 2 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mean = np.floor(0.5 * kernel_size)
        kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
        np_kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
        kernel = np_kernel / np.sum(np_kernel)
        return kernel, kernel_size

    device = input.device
    gauss_kernel, kernel_size = make_kernel(radius)
    padding = kernel_size // 2

    # Convert numpy kernel to PyTorch tensor
    gauss_kernel = torch.from_numpy(gauss_kernel).to(device)
    gauss_kernel = gauss_kernel.view(1, 1, kernel_size, kernel_size)
    gauss_kernel = gauss_kernel.repeat(input.shape[1], 1, 1, 1)

    # Apply padding
    x = F.pad(input, (padding, padding, padding, padding), mode="reflect")

    # Apply convolution
    x = F.conv2d(x, gauss_kernel, groups=input.shape[1], padding=0)

    return x
