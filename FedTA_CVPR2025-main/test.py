import os
import random
import shutil

# 设置原始文件夹和目标文件夹的路径
source_dir = 'D:\datasets\imagenet2012'
target_dir = 'D:\datasets\ILSVRC2012'

# 确保目标文件夹存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历原始文件夹中的每个文件夹
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        # 获取文件夹中所有文件的列表
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # 如果文件夹中的文件数量大于50，则随机选择50个文件
        if len(files) > 50:
            selected_files = random.sample(files, 50)
        else:
            selected_files = files

        # 创建目标文件夹（如果不存在）
        target_folder_path = os.path.join(target_dir, folder_name)
        if not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)

        # 复制选中的文件到目标文件夹
        for file_name in selected_files:
            shutil.copy(os.path.join(folder_path, file_name), target_folder_path)

print("Done!")