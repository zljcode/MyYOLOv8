import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_list = ['PR_Curvedata/ImpYOLOv8_test.csv', 'PR_Curvedata/YOLOv8_test.csv','PR_Curvedata/YOLOv5n_test.csv',
                 'PR_Curvedata/YOLOv7-tiny_PRtest.csv','PR_Curvedata/YOLOv10n.csv']
    names = ['impYOLOv8', 'YOLOv8','YOLOv5','YOLOv7','YOLOv10']
    ap = ['0.940', '0.933', '0.922', '0.811', '0.926']

    plt.figure(figsize=(6, 6))
    for i in range(len(file_list)):
        pr_data = pd.read_csv(file_list[i], header=None)
        recall, precision = np.array(pr_data[0]), np.array(pr_data[1])

        plt.plot(recall, precision, label=f'{names[i]} ap:{ap[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pr.png')