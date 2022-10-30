import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils import data
import network
import utils
from datasets.refuge import REFUGESegmentation
from datasets.data_utils import convert_img
from utils.config import get_argparser
from utils.utils import set_bn_momentum
import ttach as tta


def test(opts, model, save_dir, test_loader, test_flag):
    model.eval()

    with torch.no_grad():
        for images, img_name in test_loader:
            images = images.to(opts.device, dtype=torch.float32)
            tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
            outputs = tta_model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            for i in range(len(preds)):
                pred = preds[i]
                pred = test_loader.dataset.decode_target(pred).astype(np.uint8)
                if test_flag:
                    cv2.imwrite('./%s/%s.bmp' % (save_dir, img_name[i]), pred)
                else:
                    cv2.imwrite('./%s/%s.png' % (save_dir, img_name[i]), pred)

    convert_img(opts, save_dir)
    print('save result')

def main():
    opts = get_argparser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup random seed
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    test_transform = A.Compose([
        A.Normalize(mean=[0.6900, 0.3944, 0.1737], std=[0.2274, 0.1806, 0.1443]),
        ToTensorV2(),
    ])

    # Test_pre_mask | Test_result
    root = './data/refuge'
    save_dir = os.path.join(root, 'Test_result')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if 'pre' in save_dir:
        path = 'Test_pre_crop'
        test_flag = False
    else:
        path = 'Test_crop'
        test_flag = True

    test_images_path = os.path.join(root, path)
    test_dist = REFUGESegmentation(test_images_path, transform=test_transform, test=True)

    test_loader = data.DataLoader(
        test_dist, batch_size=opts.batch_size, shuffle=False, num_workers=16)
    print('Dataset: %s, Test set: %d' % (opts.dataset, len(test_dist)))

    # Set up model
    model_map = {
        'deeplabv3_resnet34': network.deeplabv3_resnet34,
        'deeplabv3_efficientnet_b2': network.deeplabv3_efficientnet_b2,
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    set_bn_momentum(model.backbone, momentum=0.01)

    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(opts.device)

    test(opts, model, save_dir, test_loader, test_flag)


if __name__ == '__main__':
    main()
