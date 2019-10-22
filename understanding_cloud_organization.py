import numpy as np
import pandas as pd

if __name__ == "__main__":
    # data_label = np.loadtxt('./Dataset/understanding_cloud_organization/train.csv/train.csv', dtype=np.str)
    datas = pd.read_csv('./Dataset/understanding_cloud_organization/train.csv/train.csv', sep=',', dtype='unicode')
    data_list = []
    columns = list(datas["Image_Label"])
    for idx, data in enumerate(columns):
        data_list.append([data,""])

    columns = list(datas["EncodedPixels"])
    for idx, data in enumerate(columns):
        data_list[idx][1] = data
        print('data', data_list[idx][0], data_list[idx][1])

