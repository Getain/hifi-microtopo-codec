import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
import numpy as np
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())
CUDA_VISIBLE_DEVICES=3

def compute_psnr(a, b):
    #  a, b : tensor[1,3,256,256]
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def calculate_psnr(a, b):
    """
    计算峰值信噪比 (PSNR)。

    :param image_true: 真实图像的numpy数组，浮点型，范围通常是[0, 1]。
    :param image_test: 重建图像的numpy数组，同上。
    :return: PSNR值。
    """
    image_true = a.cpu().numpy()
    image_test = b.cpu().numpy()
    # print(max(image_true))
    mse = np.mean((image_true - image_test) ** 2)
    # if mse == 0:  # 防止除以0错误
    #     return 100
    # max_pixel = 255.0  # 假设图像的最大像素值为1.0（归一化图像）
    psnr = -10 * np.log10(mse) + 20 * np.log10(255)
    return psnr

def compute_msssim(a, b):
    # 计算的是MS-SSIM的负对数形式，通常用于将MS-SSIM的值转换成一个损失函数
    # 返回值越大，则MS-SSIM对应的损失值越大，即a,b两张图片越不相似
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp_y(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    # return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    #           for likelihoods in out_net['likelihoods'].values()).item()
    likelihood = out_net['likelihoods']['y']
    # bpp = torch.count_nonzero(likelihood)*8.0 / num_pixels
    bpp = torch.log(likelihood).sum() / (-math.log(2) * num_pixels)
    return bpp

def compute_bpp_all(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def compute_mhd(A):
    # 假设A是一个形状为[b, m, w, h]的张量，其中m>1
    b, m, w, h = A.shape
    reshape_a  = A.reshape(b, m, -1)  # [b, m, w*h]

    # 确保通道数量大于1，否则无法计算相邻通道差值
    assert m > 1, "通道数量必须大于1才能计算相邻通道的差值"

    # 初始化一个张量来存储每对相邻通道的差值的绝对值
    differences = torch.zeros_like(reshape_a[:, :-1, :])  # 不包括最后一个通道

    # 计算相邻通道的差值并取绝对值
    for i in range(m - 1):  # 遍历直到倒数第二个通道
        differences[:, i, :] = torch.abs(reshape_a[:, i, :] - reshape_a[:, i + 1, :])


    # 沿着通道维度求和得到每个样本的总差值损失
    # [b, m, w*h] → [b, w*h] → [b, 1]
    loss_per_sample = differences.sum(dim=1).mean(dim=1) # 这里已经没有通道维度了，所以求和即可
    # print(loss_per_sample)
    # 对所有样本求求和得到最终损失
    final_loss = loss_per_sample.mean()
    max_mhd_loss = loss_per_sample.max()
    min_mhd_loss = loss_per_sample.min()
    return final_loss, max_mhd_loss, min_mhd_loss

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default='./ckpt/0.05checkpoint_best.pth.tar')
    parser.add_argument("--data", type=str, help="Path to dataset", default='../LIC_TCM-main/specimen_dataset/test')
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=30)
    net = net.to(device)
    net.eval()
    count = 0
    PSNR1 = 0
    PSNR2 = 0
    Bit_rate_y = 0
    Bit_rate_all = 0
    MS_SSIM = 0
    mhd_loss = 0
    max_mhd_loss = 0.0
    min_mhd_loss = float('inf')
    total_time = 0
    dictory = {}
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
    if args.real:
        net.update()
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
            x = img.unsqueeze(0)
            x_padded, padding = pad(x, p)
            count += 1
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_enc = net.compress(x_padded)
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                num_pixels = x.size(0) * x.size(2) * x.size(3)
                print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
                print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
                print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')

                Bit_rate_all += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                PSNR1 += compute_psnr(x, out_dec["x_hat"])
                MS_SSIM += compute_msssim(x, out_dec["x_hat"])
                mhd, max_mhd, min_mhd = compute_mhd(out_dec["likelihoods"]["y"])
                mhd_loss += mhd
                max_mhd_loss = max(max_mhd, max_mhd_loss)
                min_mhd_loss = min(min_mhd, min_mhd_loss)
                print(f'Mhd:{mhd:.2f}dB')
                # print('最大的mhd:', max_mhd_loss)
                # print('最小的mhd:',min_mhd_loss)


    else:
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert('RGB')
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            x_padded, padding = pad(x, p)
            count += 1
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_net = net.forward(x_padded)
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                out_net['x_hat'].clamp_(0, 1)
                out_net["x_hat"] = crop(out_net["x_hat"], padding)
                # print(f'PSNR1: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
                # print(f'PSNR2: {calculate_psnr(x, out_net["x_hat"]):.2f}dB')
                # print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB')
                # print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
                PSNR1 += compute_psnr(x, out_net["x_hat"])
                PSNR2 +=calculate_psnr(x, out_net["x_hat"])
                MS_SSIM += compute_msssim(x, out_net["x_hat"])
                Bit_rate_y += compute_bpp_y(out_net)
                Bit_rate_all += compute_bpp_all(out_net)
                mhd, max_mhd, min_mhd = compute_mhd(out_net["likelihoods"]["y"])
                mhd_loss += mhd
                max_mhd_loss = max(max_mhd, max_mhd_loss)
                min_mhd_loss = min(min_mhd, min_mhd_loss)
                # print(f'Mhd:{mhd:.2f}dB')
                # print('最大的mhd:', max_mhd_loss)
                # print('最小的mhd:', min_mhd_loss)

    PSNR1 = PSNR1 / count
    PSNR2 = PSNR2 / count
    MS_SSIM = MS_SSIM / count
    Bit_rate_y = Bit_rate_y / count
    Bit_rate_all = Bit_rate_all / count
    total_time = total_time / count
    mhd = mhd / count
    print(f'average_PSNR1: {PSNR1:.3f}dB')
    # print(f'average_PSNR2: {PSNR2:6f}dB')
    print(f'average_MS-SSIM: {MS_SSIM:.3f}')
    print(f'average_Bit-rate_y: {Bit_rate_y:.3f} bpp')
    print(f'average_Bit-rate_all: {Bit_rate_all:.3f} bpp')
    print(f'average_mhd: {mhd:.3f} ms')
    print(f'average_time: {total_time:.3f} ms')
    # print(f'最大的mhd: {max_mhd_loss:.3f}')
    # print(f'最小的mhd: {min_mhd_loss:.3f}')
    

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
    