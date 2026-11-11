import os
import random
import shutil
import tempfile
import zipfile
from PIL import Image

# 参数设置


import os
import random
import shutil
import zipfile
import tempfile
from PIL import Image

def extract_and_process_single_zip(outer_zip, inner_zip_name, output_dir, temp_dir, num):
    """解压单个内部压缩包并处理图片"""
    inner_zip_path = os.path.join(temp_dir, inner_zip_name)
    with outer_zip.open(inner_zip_name, 'r') as inner_zip_file, open(inner_zip_path, 'wb') as f:
        f.write(inner_zip_file.read())

    cropped_images = []
    with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip:
        image_names = [g for g in inner_zip.namelist() if g.endswith(('.jpg', '.png'))]
        selected_images = random.sample(image_names, min(40, len(image_names)))  # 保证不超过实际图片数量
        crop_num = 0
        for img_name in selected_images:
            with inner_zip.open(img_name) as img_file:
                img = Image.open(img_file)
                width, height = img.size
                if width >= 256 and height >= 256:
                    x = random.randint(0, width - 256)
                    y = random.randint(0, height - 256)
                    img_cropped = img.crop((x, y, x+256, y+256))
                    png_name = 'img' + f"{num:05d}" + '.jpg'
                    img_cropped.save(os.path.join(output_dir, png_name))
                    cropped_images.append(img_name)
                    num += 1
                    crop_num += 1
                if crop_num == 30:
                    print('共有',num,'张图片裁剪完成')
                    break

    os.remove(inner_zip_path)  # 清理
    return num

def process_outer_zip(outer_zip_path, output_dir, temp_dir, batch_size=1):
    num = 0
    """处理外部压缩包，分批解压内部压缩包并处理图片"""
    with zipfile.ZipFile(outer_zip_path, 'r') as outer_zip:
        inner_zips = outer_zip.namelist()
        total_inner_zips = len(inner_zips)
        processed_count = 0
        while processed_count < total_inner_zips:
            batch_end = min(processed_count + batch_size, total_inner_zips)
            current_batch = inner_zips[processed_count:batch_end]
            for inner_zip_name in current_batch:
                 num = extract_and_process_single_zip(outer_zip, inner_zip_name, output_dir, temp_dir, num)
            processed_count += batch_size
            print(f"Processed {processed_count}/{total_inner_zips} inner zips.")
    shutil.rmtree(temp_dir)
    print("All operations completed.")

if __name__ == "__main__":
    random_numbers = random.sample(range(0, 1299 + 1), 40)
    sorted_numbers = sorted(random_numbers)
    print(sorted_numbers)
    outer_zip_path = 'D:/dataset/imagenet/ImageNet/data/ImageNet2012/ILSVRC2012_img_train.tar'  # 外部压缩包路径
    output_dir = 'D:\dataset\imagenet_crop/train'  # 输出图片的目录
    temp_dir = tempfile.mkdtemp()  # 创建临时目录
    batch_size = 1  # 每次处理的外部压缩包数量，根据实际情况调整

    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(outer_zip_path, exist_ok=True)
    process_outer_zip(outer_zip_path, output_dir, temp_dir)