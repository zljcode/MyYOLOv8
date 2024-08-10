'''
此脚本用于将标注好的图像将标注框显示出来
'''
import cv2
import os

def draw_boxes(image_path, label_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    height, width, _ = image.shape

    # 读取标注文件
    if not os.path.exists(label_path):
        print(f"Error: Label file {label_path} does not exist")
        return

    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Warning: Invalid line format in {label_path}: {line}")
            continue

        class_id, x_center, y_center, box_width, box_height = map(float, parts)

        # 将相对坐标转换为绝对坐标
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        box_width = int(box_width * width)
        box_height = int(box_height * height)

        # 计算边界框的左上角和右下角坐标
        x_min = int(x_center - box_width / 2)
        y_min = int(y_center - box_height / 2)
        x_max = int(x_center + box_width / 2)
        y_max = int(y_center + box_height / 2)

        # 确保坐标是正数
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, width)
        y_max = min(y_max, height)

        # 绘制标注框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # # 绘制类别标签
        # label = f"{int(class_id)}"
        # cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 调整图像大小为640x640
    resized_image = cv2.resize(image, (640, 640))

    # 保存结果图像
    cv2.imwrite(output_path, resized_image)
    print(f"Output image saved to {output_path}")


if __name__ == "__main__":
    image_path = "/mnt/d/CV_Code/YOLOv8/ultralytics/ultralytics/datasets/HRSID_JPG/images/test/P0137_5.jpg"
    label_path = "/mnt/d/CV_Code/YOLOv8/ultralytics/ultralytics/datasets/HRSID_JPG/labels/test/P0137_5.txt"
    output_path = "/mnt/d/CV_Code/YOLOv8/ultralytics/ultralytics/My_Script/标注框图像/HRSID_P0137_5.jpg"

    draw_boxes(image_path, label_path, output_path)