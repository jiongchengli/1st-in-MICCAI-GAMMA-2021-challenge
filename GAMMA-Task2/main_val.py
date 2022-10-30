import argparse
import os
import cv2
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class customData(Dataset):
    def __init__(self, img_path, txt_path, transform=None, loader=pil_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()

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

        if self.transform is not None:
            img = self.transform(img)

        return img, pixel_size, pixel_label, img_name


def main():
    parser = argparse.ArgumentParser('argument for test')
    opt = parser.parse_args()
    opt.result_path = ''
    opt.dataset = 'final_val_loc_fine_4'
    opt.vis_name = 'final_val_predict_fine'
    opt.plain_dataset = 'val_loc'
    # imagenet / fine / coarse
    opt.normalize_flag = 'fine'

    # csv_name = 'Localization_Results-1'
    csv_name = 'res-4'
    ckpt_path = './result/best_fine_4.pth'

    model = EfficientNet.from_name('efficientnet-b4', num_classes=2)
    model.load_state_dict(torch.load(ckpt_path))

    model = model.cuda()
    cudnn.benchmark = True
    print('model weight {0} load success'.format(ckpt_path))

    image_map = dict()
    image_path = './dataset/{}/img'.format(opt.plain_dataset)
    for image_name in os.listdir(image_path):
        image = Image.open(image_path + '/' + image_name)
        image_map[image_name] = image.size

    img_root = './dataset/{}/'.format(opt.dataset)
    txt_root = './dataset/{}/'.format(opt.dataset)
    te_img_path = img_root + 'img/'
    te_txt_path = txt_root + 'val_name_loc.txt'
    vis_predict_path = 'vis_result/{}/'.format(opt.vis_name)

    if not os.path.exists(vis_predict_path):
        os.mkdir(vis_predict_path)

    affine_flag = False
    affine_dict = dict()

    if os.path.exists(txt_root + 'affine.txt'):
        affine_flag = True
        with open(txt_root + 'affine.txt') as input_file:
            lines = input_file.readlines()

        for i in range(len(lines)):
            a1, a2, a3 = lines[i].strip().split()
            affine_dict[a1] = [float(a2), float(a3)]

    if opt.normalize_flag == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        print('using imagenet mean_std')
    elif opt.normalize_flag == 'coarse':
        mean = [0.37758928, 0.18647607, 0.05481077]
        std = [0.26575511, 0.14306038, 0.06513966]
        print('using coarse mean_std')
    else:
        mean = [0.62779895, 0.30252203, 0.07727987]
        std = [0.08888372, 0.05966091, 0.03971646]
        print('using fine mean_std')

    te_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = customData(te_img_path, te_txt_path, te_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    model.eval()
    cache = []

    with torch.no_grad():
        for images, pixel_sizes, pixel_labels, image_names in test_loader:
            images = Variable(images.cuda())
            pixel_sizes = Variable(pixel_sizes.cuda())
            out = model(images)
            pixel_out = torch.mul((out + 1) / 2, pixel_sizes)

            for i in range(images.size(0)):
                # predict
                idx = image_names[i].rfind('/')
                image_name = image_names[i][idx + 1:]
                image = cv2.imread(image_names[i])
                p_x, p_y = pixel_out[i][0].item(), pixel_out[i][1].item()
                x, y = int(p_x), int(p_y)
                cv2.line(image, (x, y - 50), (x, y + 50), (0, 0, 255), 3)
                cv2.line(image, (x - 50, y), (x + 50, y), (0, 0, 255), 3)
                # print(tmp)
                cv2.imwrite(vis_predict_path + image_name, image)

                real_w, real_h = image_map[image_name][0], image_map[image_name][1]
                if affine_flag:
                    p_x += affine_dict[image_name][0]
                    p_y += affine_dict[image_name][1]
                real_x = p_x + (real_w - 2000) // 2.0
                real_y = p_y + (real_h - 2000) // 2.0

                print('{0} W:{1} H:{2}'.format(image_name, real_w, real_h))
                print('2000-Predict: {0} {1}'.format(p_x, p_y))
                print('Real-Predict: {0} {1}'.format(real_x, real_y))

                # gen result csv
                cache.append([image_name[:-4], real_x, real_y])
        submission_result = pd.DataFrame(cache, columns=['data', 'Fovea_X', 'Fovea_Y'])
        submission_result[['data', 'Fovea_X', 'Fovea_Y']].to_csv("./{}.csv".format(csv_name), index=False)


if __name__ == '__main__':
    main()
