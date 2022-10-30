import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class customData(Dataset):
    def __init__(self, img_path, txt_path, transform=None, loader=default_loader):
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

    ckpt_path = './result/best_coarse_4.pth'
    res_root = './dataset/final_val_loc_fine_4/'
    # imagenet / coarse
    normalize_flag = 'coarse'

    if not os.path.exists(res_root):
        os.mkdir(res_root)

    model = EfficientNet.from_name('efficientnet-b4', num_classes=2)
    model.load_state_dict(torch.load(ckpt_path))
    model = model.cuda()
    cudnn.benchmark = True
    print('model weight {0} load success'.format(ckpt_path))

    img_root = './dataset/val_loc_aug/'
    txt_root = './dataset/val_loc_aug/'
    te_img_path = img_root + 'img/'
    te_txt_path = txt_root + 'val_name_loc.txt'

    res_img = res_root + 'img'
    res_affine = res_root + 'affine.txt'
    res_val = res_root + 'val_name_loc.txt'

    vis_path = './vis_result/val_predict_coarse_b4'

    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    if not os.path.exists(res_img):
        os.mkdir(res_img)

    if normalize_flag == 'coarse':
        mean = [0.37758928, 0.18647607, 0.05481077]
        std = [0.26575511, 0.14306038, 0.06513966]
        print('using coarse mean_std')
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        print('using imagenet mean_std')

    te_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = customData(te_img_path, te_txt_path, te_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    model.eval()
    affine_txt = open(res_affine, "w")
    val_txt = open(res_val, "w")

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
                # vis
                vis_image = image.copy()
                cv2.rectangle(vis_image, (x-256, y-256), (x+256, y+256), (0, 0, 255), 2)
                cv2.imwrite(vis_path + '/' + image_name, vis_image)
                # gen fine image
                print('predict:', image_name, x, y)
                image = image[y - 256:y + 256, x - 256:x + 256]
                cv2.imwrite(res_img + '/' + image_name, image)
                # affine txt
                x_affine = p_x - 256
                y_affine = p_y - 256
                print('affine:', image_name, x_affine, y_affine)
                affine_txt.write("{0} {1} {2}\n".format(image_name, x_affine, y_affine))
                val_txt.write("{0} {1} {2} {3} {4}\n".format(image_name, 512, 512, -1, -1))


if __name__ == '__main__':
    main()
