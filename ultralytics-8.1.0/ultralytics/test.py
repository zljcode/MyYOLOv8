from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'yolov8-SPD-C2f_ECA-DyS-GhostC-EIoU.yaml')  # build a new model from YAML
    model.info()

