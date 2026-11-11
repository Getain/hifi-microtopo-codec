
import os
import random
import tarfile
from PIL import Image




# 定义一些变量
source_tar_path = 'D:/dataset/imagenet/ImageNet/data/ImageNet2012/ILSVRC2012_img_val.tar'  # 外部压缩包路径
output_folder = 'D:\dataset\imagenet_crop/test'  # 输出图片的目录

num_images_per_tar = 30  # 每个内部tar包中随机选取的图片数
image_size = (256, 256)  # 裁剪图片的目标尺寸
counter = 1 # 图片计数器

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 打开外部tar文件
with tarfile.open(source_tar_path, 'r') as outer_tar:
    # 随机从1300张图片选40个索引
    random_numbers = random.sample(range(0, 50000 + 1), 5500)
    sorted_numbers = sorted(random_numbers)
    # 遍历内部tar中的文件，寻找图片
    image_count = 0
    idx = 0
    for img_member in outer_tar:
        # 确保是图片文件
        if not img_member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if idx in sorted_numbers:
            # 读取并打开图片
            img_file = outer_tar.extractfile(img_member)
            img = Image.open(img_file)

            # 随机裁剪
            width, height = img.size
            if width >= 256 and height >= 256:
                x = random.randint(0, width - 256)
                y = random.randint(0, height - 256)
                cropped_img = img.crop((x, y, x + image_size[0], y + image_size[1]))

                # 保存裁剪后的图片
                output_path = os.path.join(output_folder, f'img{counter:04d}.jpg')
                cropped_img.save(output_path, format='JPEG')
                counter += 1
                image_count += 1
        idx += 1

        if image_count % 100 == 0:
            print('共有',counter-1,'张图片裁剪完成')

        # 达到每包图片处理数量后退出循环
        if image_count > 50000:

            break


print("处理完成。")