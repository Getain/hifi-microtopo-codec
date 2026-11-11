import cv2
import os
import numpy as np
from tifffile import imread, imwrite


def tif_to_png(directory, number, crop_size):
    """
    将指定目录下的所有.tif文件转换为.png文件。

    :param directory: 包含TIFF文件的目录路径
    :param number: 为当前训练集中已有的样本数量
    """
    num = number
    for filename in os.listdir(directory):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            # 读取TIFF图像
            tiff_image = imread(os.path.join(directory, filename))

            # 构建对应的PNG文件名
            png_filename = 'img'+f"{num:03d}" + '.png'
            png_path = os.path.join('/data/yujie/specimen_dataset/train_ori', png_filename)

            # 使用OpenCV将图像转换并保存为PNG格式
            cv2.imwrite(png_path, tiff_image)
            # print(f'{filename} 已成功转换为 {png_filename}')

            # 加载PNG图像以进行裁剪（这一步其实可以直接在内存中操作，但为了清晰展示步骤分开）
            png_image = cv2.imread(png_path)

            # 随机裁剪至256x256
            height, width, _ = png_image.shape

            # 确保裁剪尺寸不超过原图尺寸
            if height < crop_size[0] or width < crop_size[1]:
                raise ValueError("裁剪尺寸大于原图尺寸")

            x_start = np.random.randint(0, width - 256)
            y_start = np.random.randint(0, height - 256)
            cropped_image = png_image[y_start:y_start + 256, x_start:x_start + 256]

            # 保存裁剪后的图像
            cropped_png_path = os.path.join('/data/yujie/specimen_dataset/train', png_filename)
            cv2.imwrite(cropped_png_path, cropped_image)
            print(f"{filename} 已转换并裁剪为 {cropped_png_path}")

            num += 1
    return num


import os
from sklearn.model_selection import train_test_split
import shutil


def split_dataset(images_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    划分图像数据集为训练集、验证集和测试集。

    :param images_dir: 图片所在目录
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :param seed: 随机种子
    """
    # 获取所有图片的路径
    all_images = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
    np.random.seed(seed)  # 设置随机种子以获得可复现的结果

    # 先划分训练集和剩余集
    train, temp = train_test_split(all_images, test_size=(1 - train_ratio), random_state=seed)

    # 再从剩余集中划分验证集和测试集
    val, test = train_test_split(temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)

    # 创建相应的目录（如果不存在）
    dirs = ['train', 'val', 'test']
    for dir_name in dirs:
        dir_path = os.path.join('/data/yujie/specimen_dataset', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 移动文件到相应的目录
    for data_type, data_list in zip(dirs, [train, val, test]):
        for img_path in data_list:
            target_path = os.path.join('/data/yujie/specimen_dataset', data_type, os.path.basename(img_path))
            shutil.copy(img_path, target_path)  # 修改为复制操作
            print(f"Moved {os.path.basename(img_path)} to {data_type}/")

    print("Dataset splitting complete.")





if __name__ == "__main__":
    # directorys = '/data/yujie/dataset'  # 请将此处的 'your_directory_path' 替换为您的TIFF文件所在目录
    # num = 0
    # for directory in os.listdir(directorys):
    #     path = os.path.join(directorys, directory)
    #     num = tif_to_png(path, num, crop_size=(256,256))
    # print("训练集共有：", num)

    # 设定图片所在目录
    images_directory = '/data/yujie/specimen_dataset/dataset'  # 替换为你的图片目录
    split_dataset(images_directory)