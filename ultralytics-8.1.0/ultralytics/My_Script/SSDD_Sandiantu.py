"""
此代码脚本用于SSDD数据集目标大小散点图的生成
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 标注文件所在的文件夹路径
annotations_dir = '/mnt/d/CV_Code/YOLOv8/ultralytics/ultralytics/datasets/SSDD/biaozhu'

# 读取所有标注文件
annotations = []
for filename in os.listdir(annotations_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(annotations_dir, filename), 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:  # 确保有足够的元素
                    class_name = parts[0]
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append({'class': class_name, 'x_center': x_center, 'y_center': y_center, 'width': width, 'height': height})
                else:
                    print(f"Skipping line in file {filename} due to insufficient elements: {line}")

# 提取框的宽度和高度
widths = [ann['width'] for ann in annotations]
heights = [ann['height'] for ann in annotations]

# 创建一个DataFrame来存储宽度和高度
df = pd.DataFrame({'width_ratio': widths, 'height_ratio': heights})

# 使用Seaborn绘制散点图
plt.figure(figsize=(7.7, 7.7))  # 设置图形的尺寸为7.7x7.7英寸
sns.scatterplot(x='width_ratio', y='height_ratio', data=df, legend=False)

# 移除坐标线但保留刻度
plt.tick_params(axis='both', which='both', length=5, width=1, direction='out')  # 设置刻度线长度和方向
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# 设置x轴和y轴的标签
plt.xlabel('width', fontsize=14)  # 设置x轴标签字号为14
plt.ylabel('height', fontsize=14)  # 设置y轴标签字号为14000

# 设置x轴和y轴的范围和刻度间隔
plt.xlim(-0.05, 0.85)  # 设置x轴范围为-0.05到1.05
plt.ylim(-0.05, 0.85)  # 设置y轴范围为-0.05到0.85
plt.xticks(ticks=np.arange(0, 0.81, 0.2), fontsize=12)  # x轴刻度间隔为0.2，字号为12
plt.yticks(ticks=np.arange(0, 0.81, 0.2), fontsize=12)  # y轴刻度间隔为0.2，字号为12

# 确保线段长度相同
plt.gca().set_aspect('equal', adjustable='box')

# 调整图形布局，使其占据整个770x770像素的区域
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

# 保存图像为高DPI的文件
plt.savefig('output.png', dpi=1000, bbox_inches='tight', pad_inches=0)

plt.show()