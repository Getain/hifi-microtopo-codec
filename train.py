import argparse
import math
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import TCM
from torch.utils.tensorboard import SummaryWriter   
import os
import matplotlib as plt

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

def channel_difference_loss(A):
    """
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
    # print(max_mhd_loss)
    return final_loss



    """

def one_hot_loss(B, lambda_sparse=0.5, lambda_norm=0.5):
    b, m, w, h = B.shape
    A = B.reshape(b, m, -1)  # [b, m, w*h]
    # # 计算每行的最大值
    # max_values, _ = torch.max(A, dim=-1, keepdim=True)
    #
    # # 计算稀疏性损失
    # sparse_loss = (torch.sum(torch.abs(A), dim=-1) - max_values.squeeze(-1)).mean()
    #
    # # 计算归一化损失
    # norm_loss = (torch.sum(torch.abs(A), dim=-1) - 1).pow(2).mean()
    #
    # # 总损失
    # total_loss = lambda_sparse * sparse_loss + lambda_norm * norm_loss
    total_loss = 0.0

    return total_loss

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse', mhd_l=0.02, bpp_l=1, one_l=0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda 
        self.type = type
        self.mhd_l = mhd_l
        self.bpp_l = bpp_l
        self.one_l = one_l

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

         # 计算y_likelihoods相邻通道的曼哈顿距离，以约束相邻通道学习信息的相似性
        out["mhd_loss"] = channel_difference_loss(output["likelihoods"]["y"])

        # 计算y_likelihoods的one-hot loss
        out["onehot_loss"] = one_hot_loss(output["likelihoods"]["y"])


        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + self.bpp_l * out["bpp_loss"] + self.mhd_l * out["mhd_loss"]
            out["loss"] += out["onehot_loss"] *self.one_l
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) +  self.bpp_l * out["bpp_loss"] + self.mhd_l * out["mhd_loss"]
            out["loss"] += out["onehot_loss"] * self.one_l

        return out


def channel_difference_loss(A):
    b, m, _, _ = A.shape
    if m < 2:
        return A.new_tensor(0.0)

    A = A.reshape(b, m, -1)
    differences = (A[:, 1:, :] - A[:, :-1, :]).abs()
    return differences.sum(dim=1).mean(dim=1).mean()


def one_hot_loss(B, lambda_sparse=0.5, lambda_norm=0.5):
    return B.new_tensor(0.0)


def _channel_group_subvectors(embedding, num_groups):
    if embedding.dim() != 4:
        raise ValueError("orthogonal_loss expects a latent tensor shaped [B, C, H, W].")
    if num_groups < 2:
        raise ValueError("orthogonal_loss needs at least two sub-vectors.")
    if embedding.size(1) < num_groups:
        raise ValueError("The latent channel count is smaller than the number of groups.")

    base_width = embedding.size(1) // num_groups
    extra = embedding.size(1) % num_groups
    chunks = []
    start = 0
    for group_index in range(num_groups):
        end = start + base_width + (1 if group_index < extra else 0)
        chunks.append(embedding[:, start:end, :, :])
        start = end
    flattened = [chunk.reshape(chunk.size(0), -1) for chunk in chunks]
    max_dim = max(chunk.size(1) for chunk in flattened)
    padded = [
        F.pad(chunk, (0, max_dim - chunk.size(1))) if chunk.size(1) < max_dim else chunk
        for chunk in flattened
    ]
    return torch.stack(padded, dim=1)


def orthogonal_loss(embedding, num_groups=4, eps=1e-8):
    subvectors = _channel_group_subvectors(embedding, num_groups)
    subvectors = F.normalize(subvectors, p=2, dim=-1, eps=eps)
    gram = torch.matmul(subvectors, subvectors.transpose(1, 2))
    mask = ~torch.eye(num_groups, dtype=torch.bool, device=embedding.device)
    return gram[:, mask].pow(2).mean()


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss with four-subvector orthogonal regularization."""

    def __init__(
        self,
        lmbda=1e-2,
        type='mse',
        mhd_l=0.02,
        bpp_l=1,
        one_l=0,
        orth_l=0.1,
        orth_groups=4,
        orth_start_epoch=0,
        orth_warmup_epochs=0,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type
        self.mhd_l = mhd_l
        self.bpp_l = bpp_l
        self.one_l = one_l
        self.orth_l = orth_l
        self.orth_groups = orth_groups
        self.orth_start_epoch = orth_start_epoch
        self.orth_warmup_epochs = orth_warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_orth_weight(self):
        if self.orth_l <= 0 or self.current_epoch < self.orth_start_epoch:
            return 0.0
        if self.orth_warmup_epochs <= 0:
            return self.orth_l
        warmup_epoch = self.current_epoch - self.orth_start_epoch + 1
        warmup = min(1.0, float(warmup_epoch) / float(self.orth_warmup_epochs))
        return self.orth_l * warmup

    def forward(self, output, target, include_regularizers=True):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mhd_loss"] = channel_difference_loss(output["likelihoods"]["y"])
        out["onehot_loss"] = one_hot_loss(output["likelihoods"]["y"])
        out["orth_loss"] = orthogonal_loss(output["para"]["y"], num_groups=self.orth_groups)

        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["rd_loss"] = (
                self.lmbda * 255 ** 2 * out["mse_loss"]
                + self.bpp_l * out["bpp_loss"]
                + self.mhd_l * out["mhd_loss"]
                + out["onehot_loss"] * self.one_l
            )
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["rd_loss"] = (
                self.lmbda * (1 - out['ms_ssim_loss'])
                + self.bpp_l * out["bpp_loss"]
                + self.mhd_l * out["mhd_loss"]
                + out["onehot_loss"] * self.one_l
            )

        orth_weight = self.get_orth_weight() if include_regularizers else 0.0
        out["orth_weight"] = out["bpp_loss"].new_tensor(orth_weight)
        out["loss"] = out["rd_loss"] + out["orth_loss"] * out["orth_weight"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, type='mse'
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(tqdm(train_dataloader, desc=f"Train epoch {epoch}")):

        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        # print(d)
        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 5000 == 0:
            if type == 'mse':
                tqdm.write(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tRD loss: {out_criterion["rd_loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f'\tMhd loss: {out_criterion["mhd_loss"].item():.2f} |'
                    f'\tOrth loss: {out_criterion["orth_loss"].item():.4f} |'
                    f'\tOrth w: {out_criterion["orth_weight"].item():.4g} |'
                    
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
            else:
                tqdm.write(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tRD loss: {out_criterion["rd_loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f'\tMhd loss: {out_criterion["mhd_loss"].item():.2f} |'
                    f'\tOrth loss: {out_criterion["orth_loss"].item():.4f} |'
                    f'\tOrth w: {out_criterion["orth_weight"].item():.4g} |'
                    
                    f"\tAux loss: {aux_loss.item():.2f}"
                )


def test_epoch(epoch, test_dataloader, model, criterion, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()
        mhd_loss = AverageMeter()
        orth_loss = AverageMeter()
        # onehot_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d, include_regularizers=False)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
                mhd_loss.update(out_criterion["mhd_loss"])
                orth_loss.update(out_criterion["orth_loss"])
                # onehot_loss.update(out_criterion["onehot_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tMhd loss: {mhd_loss.avg:.2f} |"
            f"\tOrth loss: {orth_loss.avg:.4f} |"
           
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()
        mhd_loss = AverageMeter()
        orth_loss = AverageMeter()
        # onehot_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d, include_regularizers=False)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])
                mhd_loss.update(out_criterion["mhd_loss"])
                orth_loss.update(out_criterion["orth_loss"])
                # onehot_loss.update(out_criterion["onehot_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tMhd loss: {mhd_loss.avg:.2f} |"
            f"\tOrth loss: {orth_loss.avg:.4f} |"
            
            f"\tAux loss: {aux_loss.avg:.2f}\t"

        )

    return loss.avg


def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, help="Training dataset",
        default='../LIC_TCM-main/imagenet'
        # default='imagenet'
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Weight for the four-subvector orthogonal loss (default: %(default)s)",
    )
    parser.add_argument(
        "--orth-groups",
        type=int,
        default=4,
        help="Number of latent sub-vectors used by the decoder and orthogonal loss (default: %(default)s)",
    )
    parser.add_argument(
        "--orth-start-epoch",
        type=int,
        default=0,
        help="Epoch before enabling orthogonal regularization (default: %(default)s)",
    )
    parser.add_argument(
        "--orth-warmup-epochs",
        type=int,
        default=0,
        help="Epochs used to warm up orthogonal loss weight (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int, default=[45, 48]
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "tensorboard/")

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )


    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = TCM(
        config=[2,2,2,2,2,2],
        head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0.0,
        N=args.N,
        M=args.M,
        latent_groups=args.orth_groups,
    )
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch

    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion = RateDistortionLoss(
        lmbda=args.lmbda,
        type=type,
        mhd_l=args.beta,
        orth_l=args.alpha,
        orth_groups=args.orth_groups,
        orth_start_epoch=args.orth_start_epoch,
        orth_warmup_epochs=args.orth_warmup_epochs,
    )

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        criterion.set_epoch(epoch)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            type
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, type)
        writer.add_scalar('test_loss', loss, epoch)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])

# python eval.py --checkpoint '/data/yujie/pretrained/0.05.pth.tar' --data '/data/yujie/kodak' --cuda

# CUDA_VISIBLE_DEVICES='1' python -u ./train.py -d "/data/yujie/dataset" \
#     --cuda --N 64 --lambda 0.05 --epochs 50 --lr_epoch 45 48 \
#     --save_path "/data/yujie/save/save_path" --save \
#     --checkpoint "/data/yujie/save/checkpoint"

# 本地连接远程服务器tensorboard
# tensorboard --logdir="logs" --port 6006
# ssh -L 16006:127.0.0.1:6006 yujie@121.48.163.136 # @前后是你远程服务器的用户名和ip
