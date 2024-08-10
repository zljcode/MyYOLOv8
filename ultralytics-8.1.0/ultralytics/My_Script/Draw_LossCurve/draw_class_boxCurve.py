'''此代码用于画class_loss曲线并以高分辨率保存，使用系统默认衬线字体'''
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from matplotlib import rcParams

# 设置全局字体为默认衬线字体
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times New Roman', 'Times', 'Palatino', 'Charter', 'serif']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义 CSV 文件路径模式
file_pattern = '../table/*.csv'  # 替换为你的 CSV 文件路径模式

# 初始化图表
plt.figure(figsize=(10, 6))

# 循环遍历每个 CSV 文件并绘制损失曲线
for file_path in glob.glob(file_pattern):
    data = pd.read_csv(file_path)
    # 从列名中去除开头/结尾的空格
    data.columns = data.columns.str.strip()
    epochs = data['epoch']
    box_loss = data['val/dfl_loss']

    # 从文件名中提取 IoU 类型
    iou_type = os.path.basename(file_path).split('_')[0].replace('The', '')

    plt.plot(epochs, box_loss, label=f'{iou_type}')

# 设置图表属性
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Validation Dfl Loss over Epochs', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

# 设置刻度标签的字体大小
plt.tick_params(axis='both', which='major', labelsize=10)

# 保存图像
plt.savefig('Validation_Box_Loss.png', dpi=600, bbox_inches='tight')

# 显示图像（可选）
plt.show()