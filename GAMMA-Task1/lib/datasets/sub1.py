from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import torchvision.transforms as trans

from PIL import Image

import pdb

class GAMMA_sub1_dataset(Dataset):
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 img_transforms,
                 oct_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")

        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))

        # fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        # print('-----')
        # print(fundus_img.shape)  ###h,w
        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)

        fundus_img = Image.open(fundus_img_path).convert('RGB')
        # oct_series_0 = Image.open(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0])).convert('L')
        # pdb.set_trace()
        # print(fundus_img.size)  ###w,h

        # oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1]), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)#[..., np.newaxis]
            # print(oct_img[k].shape)
            # oct_img[k] = Image.open(
            #     os.path.join(self.dataset_root, real_index, real_index, p)).convert('L')[..., np.newaxis]

        # print('----')
        # print(oct_img.shape)

        oct_img = oct_img.transpose(2, 1, 0) ####ljc

        if self.img_transforms is not None:
            fundus_img = self.img_transforms(fundus_img)
        if self.oct_transforms is not None:
            # print(type(oct_img))  ###<class 'numpy.ndarray'>
            oct_img = self.oct_transforms(oct_img)

        # print(oct_img.size())

        # normlize on GPU to save CPU Memory and IO consuming.
        # fundus_img = (fundus_img / 255.).astype("float32")
        # oct_img = (oct_img / 255.).astype("float32")

        # pdb.set_trace()
        # fundus_img = fundus_img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        # oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W

        if self.mode == 'test':
            return fundus_img, oct_img, real_index
        if self.mode == "train":
            label = label.argmax()
            return fundus_img, oct_img, label

    def __len__(self):
        return len(self.file_list)


class GAMMA_sub1_dataset_512(Dataset):  #### oct_image[150:662,:]
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 img_transforms,
                 oct_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")

        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))

        # fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        # print('-----')
        # print(fundus_img.shape)  ###h,w
        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)

        fundus_img = Image.open(fundus_img_path).convert('RGB')
        # oct_series_0 = Image.open(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0])).convert('L')
        # pdb.set_trace()
        # print(fundus_img.size)  ###w,h

        # oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")
        # oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1]), dtype="uint8")
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[1], oct_series_0.shape[1]), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[150:662,:]#[..., np.newaxis]
            # print(oct_img[k].shape)
            # oct_img[k] = Image.open(
            #     os.path.join(self.dataset_root, real_index, real_index, p)).convert('L')[..., np.newaxis]

        # print('----')
        # print(oct_img.shape)

        oct_img = oct_img.transpose(2, 1, 0) ####ljc

        if self.img_transforms is not None:
            fundus_img = self.img_transforms(fundus_img)
        if self.oct_transforms is not None:
            # print(type(oct_img))  ###<class 'numpy.ndarray'>
            oct_img = self.oct_transforms(oct_img)

        # print(oct_img.size())

        # normlize on GPU to save CPU Memory and IO consuming.
        # fundus_img = (fundus_img / 255.).astype("float32")
        # oct_img = (oct_img / 255.).astype("float32")

        # pdb.set_trace()
        # fundus_img = fundus_img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        # oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W

        if self.mode == 'test':
            return fundus_img, oct_img, real_index
        if self.mode == "train":
            label = label.argmax()
            return fundus_img, oct_img, label

    def __len__(self):
        return len(self.file_list)


class GAMMA_sub1_dataset_512_cup(Dataset):  #### oct_image[150:662,:]
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 img_transforms,
                 oct_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        # fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        # fundus_img_path = os.path.join('/home3/ljc/datasets/GAMMA_dataset/crop', real_index + "_crop.jpg")
        fundus_img_path = os.path.join('E:/dataset/GAMMA_training data/training_data/2d/crop', real_index + "_crop.jpg")

        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))

        # fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        # print('-----')
        # print(fundus_img.shape)  ###h,w
        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)

        fundus_img = Image.open(fundus_img_path).convert('RGB')
        # oct_series_0 = Image.open(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0])).convert('L')
        # pdb.set_trace()
        # print(fundus_img.size)  ###w,h

        # oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")
        # oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1]), dtype="uint8")
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[1], oct_series_0.shape[1]), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[150:662,:]#[..., np.newaxis]
            # print(oct_img[k].shape)
            # oct_img[k] = Image.open(
            #     os.path.join(self.dataset_root, real_index, real_index, p)).convert('L')[..., np.newaxis]

        # print('----')
        # print(oct_img.shape)

        oct_img = oct_img.transpose(2, 1, 0) ####ljc

        if self.img_transforms is not None:
            fundus_img = self.img_transforms(fundus_img)
        if self.oct_transforms is not None:
            # print(type(oct_img))  ###<class 'numpy.ndarray'>
            oct_img = self.oct_transforms(oct_img)

        # print(oct_img.size())

        # normlize on GPU to save CPU Memory and IO consuming.
        # fundus_img = (fundus_img / 255.).astype("float32")
        # oct_img = (oct_img / 255.).astype("float32")

        # pdb.set_trace()
        # fundus_img = fundus_img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        # oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W

        if self.mode == 'test':
            return fundus_img, oct_img, real_index
        if self.mode == "train":
            label = label.argmax()
            return fundus_img, oct_img, label

    def __len__(self):
        return len(self.file_list)


class GAMMA_sub1_dataset_224(Dataset):  #### oct_image[150:662,:]
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 img_transforms,
                 oct_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")

        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))

        # fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        # print('-----')
        # print(fundus_img.shape)  ###h,w
        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)

        fundus_img = Image.open(fundus_img_path).convert('RGB')
        # oct_series_0 = Image.open(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0])).convert('L')
        # pdb.set_trace()
        # print(fundus_img.size)  ###w,h

        # oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")
        # oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1]), dtype="uint8")
        oct_img = np.zeros((len(oct_series_list), 224, 224), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            crop_img = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[150:662,:]#[..., np.newaxis]
            resize_img = cv2.resize(crop_img,(224,224))
            oct_img[k] = resize_img

            # print(oct_img[k].shape)
            # oct_img[k] = Image.open(
            #     os.path.join(self.dataset_root, real_index, real_index, p)).convert('L')[..., np.newaxis]

        # print('----')
        # print(oct_img.shape)

        oct_img = oct_img.transpose(2, 1, 0) ####ljc

        if self.img_transforms is not None:
            fundus_img = self.img_transforms(fundus_img)
        if self.oct_transforms is not None:
            # print(type(oct_img))  ###<class 'numpy.ndarray'>
            oct_img = self.oct_transforms(oct_img)

        # print(oct_img.size())

        # normlize on GPU to save CPU Memory and IO consuming.
        # fundus_img = (fundus_img / 255.).astype("float32")
        # oct_img = (oct_img / 255.).astype("float32")

        # pdb.set_trace()
        # fundus_img = fundus_img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        # oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W

        if self.mode == 'test':
            return fundus_img, oct_img, real_index
        if self.mode == "train":
            label = label.argmax()
            return fundus_img, oct_img, label

    def __len__(self):
        return len(self.file_list)

if __name__=='__main__':
    trainset_root = r'E:\dataset\GAMMA_training data\training_data\multi-modality_images'

    val_ratio = 0.2
    filelists = os.listdir(trainset_root)
    train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=42)
    print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))


    image_size = 224 # 256

    img_train_transforms = trans.Compose([
        # trans.CenterCrop((2000,2000)),
        trans.RandomResizedCrop(
            image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticalFlip(),
        trans.RandomRotation(30),
        trans.ToTensor() ##
    ])



    oct_train_transforms = trans.Compose([
        trans.ToTensor(), ###LJC
        # trans.CenterCrop(512),   ####[256] + oct_img_size   oct_img_size
        # trans.RandomHorizontalFlip(),
        # trans.RandomVerticalFlip()
    ])

    img_val_transforms = trans.Compose([
        trans.CenterCrop((2000, 2000)),  ###v2c
        trans.Resize((224, 224)),
        trans.ToTensor(),
    ])

    oct_val_transforms = trans.Compose([
        trans.ToTensor(), ###LJC
        # trans.CenterCrop(512),   ####[256] + oct_img_size   oct_img_size
        # trans.RandomHorizontalFlip(),
        # trans.RandomVerticalFlip()
    ])



    train_dataset = GAMMA_sub1_dataset_224(dataset_root=trainset_root,
                            img_transforms=img_val_transforms,
                            oct_transforms=oct_val_transforms,
                            label_file='E:/dataset/GAMMA_training data/training_data/glaucoma_grading_training_GT.xlsx')

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle = True#False
    )

    # for fundus_img, oct_img, label in train_loader:
    #     print(fundus_img.size())
    #     print(oct_img.size())
    #     print(label)


    plt.figure(figsize=(15, 5))

    for i in range(5):
        fundus_img, oct_img, lab = train_dataset.__getitem__(i)
        print(fundus_img.size())
        print(oct_img.size())
        print(lab)
        # print(fundus_img)
        # print(oct_img)
        plt.subplot(2, 5, i+1)
        # plt.imshow(fundus_img)
        plt.imshow(fundus_img.transpose(2, 0))
        plt.axis("off")
        plt.subplot(2, 5, i+6)
        plt.imshow(oct_img[100], cmap='gray')
        plt.axis("off")

    plt.show()