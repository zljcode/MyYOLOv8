"""此脚本用于将HRSID的类别id全部从1改为0"""
import os
from tqdm import tqdm


def modify_first_column(folder_path):
    # 获取文件夹中的所有txt文件
    txt_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.txt')]

    # 遍历文件夹中的所有txt文件并显示进度条
    for filename in tqdm(txt_files, desc="Processing files"):
        file_path = os.path.join(folder_path, filename)
        # 读取文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 修改第一列中的1为0
        modified_lines = []
        for line in lines:
            parts = line.split()
            if parts[0] == '1':
                parts[0] = '0'
            modified_line = ' '.join(parts) + '\n'
            modified_lines.append(modified_line)

        # 将修改后的内容写回文件
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)


# 指定文件夹路径
folder_path = '/mnt/d/CV_Code/YOLOv8/ultralytics/ultralytics/datasets/HRSID_JPG/labels1/test'
modify_first_column(folder_path)