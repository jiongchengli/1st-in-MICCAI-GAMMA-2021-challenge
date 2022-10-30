import argparse
import copy
import datetime
import os
import random

import cv2 as cv
import numpy as np
import pandas as pd
import skimage
import torch
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class customData(Dataset):
    def __init__(self, phase, img_path, txt_path, transform=None, loader=pil_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()

        print('customData - phase:', phase)

        self.phase = phase
        self.img_name = []
        self.x_size = []
        self.y_size = []
        self.x_label = []
        self.y_label = []

        for i in range(len(lines)):
            a1, a2, a3, a4, a5 = lines[i].strip().split()

            self.img_name.append(img_path + a1)
            self.x_size.append(float(a2))
            self.y_size.append(float(a3))
            self.x_label.append(float(a4))
            self.y_label.append(float(a5))

        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        img = self.loader(img_name)
        pixel_size = torch.Tensor([self.x_size[item], self.y_size[item]])
        pixel_label = torch.Tensor([self.x_label[item], self.y_label[item]])

        if self.phase == 'train':
            x_affine, y_affine, img = affine_img(img)
            pixel_label[0] += x_affine
            pixel_label[1] += y_affine

        if self.transform is not None:
            img = self.transform(img)

        gaussian_flag = random.uniform(0, 1)
        if gaussian_flag > 0.6 and self.phase == 'train':
            img = gaussian_img(img)

        return img.float(), pixel_size, pixel_label, img_name


def affine_img(pil_image):
    img = cv.cvtColor(np.asarray(pil_image), cv.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    x_affine = random.uniform(-0.1 * w, 0.1 * w)
    y_affine = random.uniform(-0.1 * h, 0.1 * h)
    M = np.float32([[1, 0, x_affine], [0, 1, y_affine]])
    img = cv.warpAffine(img, M, (w, h))
    img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    return x_affine, y_affine, img


def gaussian_img(tensor_image):
    tensor_image = np.transpose(tensor_image, (1, 2, 0))
    tensor_image = skimage.util.random_noise(tensor_image, mode='gaussian', seed=None, clip=True,
                                             var=random.uniform(0.01, 0.05))
    tensor_image = transforms.ToTensor()(tensor_image)
    return tensor_image


def train(model, train_loader, optimizer, epoch):
    model.train()

    for idx, (img, pixel_size, pixel_label, img_name) in enumerate(train_loader):

        img = Variable(img.cuda())
        pixel_size = Variable(pixel_size.cuda())
        pixel_label = Variable(pixel_label.cuda())
        out = model(img)
        pixel_out = torch.mul((out + 1) / 2, pixel_size)

        pixel_loss = torch.tensor(0.0)
        for i in range(img.size(0)):
            new_loss = torch.dist(pixel_out[i], pixel_label[i])
            pixel_loss = pixel_loss + new_loss

        pixel_loss = pixel_loss / img.size(0)
        optimizer.zero_grad()
        pixel_loss.backward()
        optimizer.step()

        print('Train: [{0}][{1}/{2}]\t'
              'Loss: {3:.2f}\t'
              .format(epoch, idx + 1, len(train_loader), pixel_loss))


def validate(model, test_loader, epoch):
    model.eval()

    max_eval_loss = 0.0
    min_eval_loss = 1000.0
    pixel_eval_loss = 0.0
    eval_num = 0

    difficult_map = dict()

    with torch.no_grad():
        for img, pixel_size, pixel_label, img_name in test_loader:
            img = Variable(img.cuda())
            pixel_size = Variable(pixel_size.cuda())
            pixel_label = Variable(pixel_label.cuda())
            out = model(img)
            pixel_out = torch.mul((out + 1) / 2, pixel_size)

            for i in range(img.size(0)):
                new_loss = torch.dist(pixel_out[i], pixel_label[i]).item()

                difficult_map[img_name] = new_loss

                pixel_eval_loss = pixel_eval_loss + new_loss
                eval_num = eval_num + 1

                if new_loss > max_eval_loss:
                    max_eval_loss = new_loss
                if new_loss < min_eval_loss:
                    min_eval_loss = new_loss

    pixel_eval_loss = pixel_eval_loss / eval_num

    print('Epoch:{}, Eval Loss: {:.2f}, '
          'Min Eval Loss: {:.2f}, Max Eval Loss: {:.2f}'.
          format(epoch, pixel_eval_loss,
                 min_eval_loss, max_eval_loss))

    difficult_map = sorted(difficult_map.items(), key=lambda d: d[1], reverse=True)
    difficult_img = ""
    for i in range(5):
        difficult_img += str(difficult_map[i]) + " "
    print('difficult_img: ' + difficult_img)

    return pixel_eval_loss, min_eval_loss, max_eval_loss


def main():
    note = '4_b4 epoch_1000 lr_1e-4 coarse'
    parser = argparse.ArgumentParser('argument for training')
    opt = parser.parse_args()

    opt.epochs = 1000
    opt.learning_rate = 1e-4
    opt.batch = 4
    opt.target = 4

    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    mean = [0.37758928, 0.18647607, 0.05481077]
    std = [0.26575511, 0.14306038, 0.06513966]

    tr_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    te_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    img_root = './dataset/train_loc_aug/'
    txt_root = './dataset/train_loc_aug/'
    tr_img_path = img_root + 'img/'
    tr_txt_path = txt_root + 'train_name_loc_{}.txt'.format(opt.target)
    te_img_path = img_root + 'img/'
    te_txt_path = txt_root + 'test_name_loc_{}.txt'.format(opt.target)

    opt.result_path = './result/'
    y = datetime.datetime.now().year
    m = datetime.datetime.now().month
    d = datetime.datetime.now().day
    cur_time = str(y) + '.' + str(m) + '.' + str(d)
    opt.result_path += cur_time
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    seed = str(random.randint(100, 999))
    opt.result_path += '/' + seed
    opt.result_path += '_epoch_{0}_lr_{1}_bsz_{2}'.format(opt.epochs, opt.learning_rate, opt.batch)
    opt.result_path += '/'

    train_data = customData('train', tr_img_path, tr_txt_path, tr_transform)
    test_data = customData('test', te_img_path, te_txt_path, te_transform)

    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    best_eval_loss = 1000.0
    best_min_eval_loss = 1000.0
    best_max_eval_loss = 0.0
    best_model = model.state_dict()
    last_model = model.state_dict()
    best_epoch = -1
    results = {'avg_loss': [], 'min_loss': [], 'max_loss': []}

    for e in range(1, opt.epochs + 1):
        train(model, train_loader, optimizer, e)

        pixel_eval_loss, min_eval_loss, max_eval_loss = validate(model, test_loader, e)

        results['avg_loss'].append(pixel_eval_loss)
        results['min_loss'].append(min_eval_loss)
        results['max_loss'].append(max_eval_loss)

        last_model = model.state_dict()

        if e > opt.epochs * 2 // 3 and pixel_eval_loss < best_eval_loss:
            best_eval_loss = pixel_eval_loss
            best_min_eval_loss = min_eval_loss
            best_max_eval_loss = max_eval_loss
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = e

    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)

    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, opt.epochs + 1))
    # print(data_frame)
    data_frame.to_csv('{}/best_loss_{}.csv'.format(opt.result_path, best_eval_loss),
                      index_label='epoch')

    # save model
    path = opt.result_path + 'best_model_{0}.pth'.format(best_epoch)
    torch.save(best_model, path)
    path = opt.result_path + 'last_model.pth'
    torch.save(last_model, path)
    print('Best Epoch:{}, Best Loss: {:.2f}, '
          'Best Min Loss: {:.2f}, Best Max Loss: {:.2f}'.
          format(best_epoch, best_eval_loss, best_min_eval_loss, best_max_eval_loss))

    print('note:', note)
    print(opt)


if __name__ == '__main__':
    main()
