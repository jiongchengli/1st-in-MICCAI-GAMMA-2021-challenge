import os
import random
import time
import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils import data
import network
import pandas as pd
import copy
from datasets.refuge import REFUGESegmentation
from datasets.data_utils import convert_img
from metrics.stream_metrics import StreamSegMetrics
from utils.utils import set_bn_momentum, AverageMeter, set_optimizer_scheduler
from utils.losses import Dice_and_CE_loss
from utils.config import get_argparser
from collections import OrderedDict


def get_dataset(opts):
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.Normalize(mean=[0.6900, 0.3944, 0.1737], std=[0.2274, 0.1806, 0.1443]),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Normalize(mean=[0.6900, 0.3944, 0.1737], std=[0.2274, 0.1806, 0.1443]),
        ToTensorV2(),
    ])

    train_images_path = os.path.join("data", opts.dataset, "Train_crop")
    valid_images_path = os.path.join("data", opts.dataset, "Valid_crop")

    train_dist = REFUGESegmentation(train_images_path, transform=train_transform)
    val_dist = REFUGESegmentation(valid_images_path, transform=val_transform)

    return train_dist, val_dist


def train(opts, model, optimizer, criterion, epoch, train_loader):
    model.train()
    losses = AverageMeter()
    idx = 1
    for (images, labels, _) in train_loader:
        images = images.to(opts.device, dtype=torch.float32)
        labels = labels.to(opts.device, dtype=torch.long)

        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.update(loss.item(), opts.batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print("Train: [{}][{}/{}] \t Loss: {:.5f}".format(epoch, idx, len(train_loader), losses.avg))

        idx += 1
    return losses.avg


def valid(opts, model, val_loader, save_dir, metrics):
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for images, labels, img_name in val_loader:
            images = images.to(opts.device, dtype=torch.float32)
            labels = labels.to(opts.device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
            if opts.valid:
                for i in range(len(preds)):
                    pred = preds[i]
                    pred = val_loader.dataset.decode_target(pred).astype(np.uint8)
                    cv2.imwrite('{0}/{1}.png'.format(save_dir, img_name[i]), pred)

        score = metrics.get_results()

    if opts.valid:
        convert_img(opts, save_dir)
    return score


def save_ckpt(state_dict, ckpt_path):
    """ save model
    """
    torch.save({
        "state_dict": state_dict
    }, ckpt_path)
    print("Model saved as {}".format(ckpt_path))


def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup random seed
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # Valid_result
    root = './data/refuge'
    save_dir = os.path.join(root, 'Valid_result')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    ckpt_path = 'checkpoints'
    metrics = StreamSegMetrics(opts.num_classes)

    ckpt_name = '{0}_bsz_{1}_lr_{2}_loss_{3}'.format(opts.model, opts.batch_size, opts.lr, opts.loss_type)
    ckpt_path = os.path.join(ckpt_path, 'cup_disc', 'os_{}'.format(opts.output_stride))
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    train_dist, val_dist = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dist, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dist, batch_size=opts.batch_size, shuffle=False, num_workers=4)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dist), len(val_dist)))

    # Set up model
    model_map = {
        'deeplabv3_resnet34': network.deeplabv3_resnet34,
        'deeplabv3_efficientnet_b2': network.deeplabv3_efficientnet_b2,
    }

    model = model_map[opts.model](
        num_classes=opts.num_classes, output_stride=opts.output_stride)
    set_bn_momentum(model.backbone, momentum=0.01)

    # Set up optimizer
    optimizer, scheduler = set_optimizer_scheduler(opts, model)

    # Set up criterion
    if opts.loss_type == 'dc_ce':
        criterion = Dice_and_CE_loss()
    else:
        criterion = nn.CrossEntropyLoss()

    print('loss type:', opts.loss_type)

    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        state_dict = model.state_dict()
        print("{} load...".format(opts.model))
        for k, v in checkpoint['state_dict'].items():
            if k in state_dict.keys():
                new_state_dict[k] = v
            else:
                print(k)
        state_dict.update(new_state_dict)
        model.load_state_dict(state_dict)
        model.to(opts.device)
        print("Model restored from {}".format(opts.ckpt))
        del checkpoint
    else:
        print("{model} start train...".format(model=opts.model))
        model.to(opts.device)

    # ========== Valid Start ==========#
    if opts.valid:
        val_score = valid(opts, model, val_loader, save_dir, metrics)
        print(metrics.to_str(val_score))
        return

    # ===== Train Start =====
    best_mIou_1 = 0.0
    best_state_dict_1 = None
    best_mIou_2 = 0.0
    best_state_dict_2 = None
    results = {'train_loss': [], 'Disc IoU': [], 'Cup IoU': [], 'Mean IoU': []}
    results_dir = os.path.join('result', 'cup_disc', 'os_' + str(opts.output_stride))

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    for epoch in range(1, opts.epochs + 1):

        time1 = time.time()
        loss = train(opts, model, optimizer, criterion, epoch, train_loader)
        time2 = time.time()
        print('Epoch {}, Total time {:.3f}'.format(epoch, time2 - time1))

        val_score = valid(opts, model, val_loader, save_dir, metrics)
        print(metrics.to_str(val_score))
        cur_mIou = val_score['Mean IoU']
        cur_state_dict = copy.deepcopy(model.state_dict())

        # 暂时根据MIoU选择最佳模型
        low_epoch = opts.epochs // 2
        mid_epoch = opts.epochs * 3 // 4
        high_epoch = opts.epochs
        if cur_mIou >= best_mIou_1 and low_epoch < epoch <= mid_epoch:
            best_mIou_1 = cur_mIou
            best_state_dict_1 = cur_state_dict

        if cur_mIou >= best_mIou_2 and mid_epoch < epoch <= high_epoch:
            best_mIou_2 = cur_mIou
            best_state_dict_2 = cur_state_dict

        results['train_loss'].append(loss)

        results['Disc IoU'].append(val_score['Disc IoU'])
        results['Cup IoU'].append(val_score['Cup IoU'])
        results['Mean IoU'].append(val_score['Mean IoU'])

        scheduler.step()

    save_ckpt(best_state_dict_1, '{0}/best_{1}_{2}.pth'.format(ckpt_path, ckpt_name, best_mIou_1))
    save_ckpt(best_state_dict_2, '{0}/best_{1}_{2}.pth'.format(ckpt_path, ckpt_name, best_mIou_2))

    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, opts.epochs + 1))
    data_frame.to_csv('{0}/best_{1}_{2}_{3}.csv'.format(results_dir, ckpt_name, best_mIou_1, best_mIou_2),
                      index_label='epoch')


if __name__ == '__main__':
    main()
