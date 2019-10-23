import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Model
from keras.losses import *
from keras.layers import Input, Conv2D, GlobalAveragePooling2D
from keras.utils import Sequence
from keras.metrics import *
from keras.applications import ResNet50


def rle2mask(mask_rle, shape=(2100, 1400)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    # s = mask_rle.split(' ')
    maskes = []
    for s in mask_rle:

        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        if s is not None:
            starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
            starts -= 1
            ends = starts + lengths
            for lo, hi in zip(starts, ends):
                img[lo:hi] = 1

        img_t = img.reshape(shape).T
        maskes.append(img_t)

    maskes = np.asarray(maskes)
    return maskes


class DataGenerator(Sequence):
    def __init__(self, X, Y, batch_size, dim, channel, image_path, shuffle=True):
        self.X = X
        self.y = Y
        self.dim = dim
        self.channel = channel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_path = image_path
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))

    def __data_generation(self, X_list, y_list):
        X = np.empty((self.batch_size, *self.dim, self.channel))

        # y = np.empty((self.batch_size, 4, *self.dim, 1), dtype=float)
        y = np.empty((self.batch_size,4), dtype=float)

        if y is not None:
            for i, (img, label) in enumerate(zip(X_list, y_list)):
                read_image = plt.imread(self.image_path+img)/255.
                X[i] = cv2.resize(read_image, (256, 256))

                for j in range(4):
                    if label[j] is None:
                        y[i,j] = 0
                    else:
                        y[i,j] = 1
                # X[i] = np.resize(read_image, (256,256,3))
                # maskes = rle2mask(label)
                # for j in range(4):
                #     mask = cv2.resize(maskes[j], (256,256))
                #     mask = np.resize(mask,(256,256,1))
                #     y[i, j] = mask

            return X, y
        else:
            for i, img in enumerate(X_list):
                X[i] = img
            return X

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        X_list = [self.X[k]for k in indexes]

        if self.y is not None:
            y_list = [self.y[k]for k in indexes]
            X, y = self.__data_generation(X_list, y_list)
            return X, y
        else:
            y_list = None
            X = self.__data_generation(X_list, y_list)
            return X

def data_load(path):
    input_data = pd.read_csv(path, sep=',', dtype='unicode')
    data_list = []
    X = []
    y = []
    img_columns = list(input_data["Image_Label"])
    pixel_columns = list(input_data["EncodedPixels"])
    length = len(img_columns)//4

    for i in range(length):
        data = img_columns[i*4].split('_')

        if not pd.isna(pixel_columns[i*4]):
            pixel_list_fish = pixel_columns[i*4].split(' ')
        else:
            pixel_list_fish = None

        if not pd.isna(pixel_columns[i*4+1]):
            pixel_list_flower = pixel_columns[i*4+1].split(' ')
        else:
            pixel_list_flower = None

        if not pd.isna(pixel_columns[i*4+2]):
            pixel_list_gravel = pixel_columns[i*4+2].split(' ')
        else:
            pixel_list_gravel = None

        if not pd.isna(pixel_columns[i*4+3]):
            pixel_list_sugar = pixel_columns[i*4+3].split(' ')
        else:
            pixel_list_sugar = None

        X.append(data[0])
        y.append([pixel_list_fish, pixel_list_flower, pixel_list_gravel, pixel_list_sugar])

        data_list.append([data[0], ])

    return X, y


def data_augmentation(data_list):
    print("H")


class TestModel:
    def __init__(self):
        input_layer = Input((256,256,3))
        regnet = ResNet50(input_tensor=input_layer, include_top=False, weights='imagenet')
        regnet_output = regnet.output
        conv = Conv2D(filters=4, kernel_size=1, strides=1, padding='same', activation='relu')(regnet_output)
        global_avg_pool = GlobalAveragePooling2D()(conv)
        self.model = Model(inputs=input_layer, outputs=global_avg_pool)
        self.model.summary()


if __name__ == "__main__":
    root_path = './Dataset/understanding_cloud_organization'
    image_path = root_path+'/train_images/'
    X, y = data_load(root_path+'/train_csv/train.csv')
    train_x = X[:-320]
    train_y = y[:-302]
    test_x = X[-320:]
    test_y = y[-320:]
    # print(len(data_list[:]))
    # print(data_list[0][:])
    train_generator = DataGenerator(train_x, train_y, 32, (256,256), 3, image_path)
    test_generator = DataGenerator(test_x, test_y, 32, (256,256), 3, image_path)
    test_model = TestModel()
    test_model.model.compile(optimizer='adam', loss=binary_crossentropy, metrics=[binary_accuracy])

    test_model.model.fit_generator(train_generator, steps_per_epoch=1000, epochs=10,
                                   validation_data=test_generator, validation_steps=10)

