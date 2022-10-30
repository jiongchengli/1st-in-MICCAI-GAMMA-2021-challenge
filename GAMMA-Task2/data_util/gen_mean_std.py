import numpy as np
import cv2
import os

mean = [0, 0, 0]
std = [0, 0, 0]

index = 1
img_num = 0
path = '../dataset/train_loc_fine/img_0'
for img_name in os.listdir(path):
    img_num += 1
    img = cv2.imread(path + '/' + img_name)
    img = np.asarray(img)
    img = img.astype(np.float64) / 255.
    for i in range(3):
        mean[i] += img[:, :, i].mean()
        std[i] += img[:, :, i].std()


mean.reverse()
std.reverse()

mean = np.asarray(mean) / img_num
std = np.asarray(std) / img_num

print("normMean = {}".format(mean))
print("normStd = {}".format(std))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(mean, std))
