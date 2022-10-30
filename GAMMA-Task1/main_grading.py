import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from lib.datasets.sub1 import GAMMA_sub1_dataset, GAMMA_sub1_dataset_512, GAMMA_sub1_dataset_512_cup, \
    GAMMA_sub1_dataset_224
import torchvision.transforms as trans
from torch.utils.data import DataLoader, Dataset
from lib.model.dual_resnet import Dual_network34, Dual_network18, Dual_networkx50
import torchvision.models as models
import pdb
from torch.autograd import Variable
import torch
import time
from lib.model.utils.config import get_argparser
import random
import skimage
from lib.losses.Focal_loss import FocalLoss
from lib.model.dual_efficientnet import Dual_efficientnetb0, Dual_efficientnetb1, Dual_efficientnetb2, \
    Dual_efficientnetb3, Dual_efficientnetb4, Dual_efficientnetb5, Dual_efficientnetb6, Dual_efficientnetb7


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)


def train(args, model, iters, train_loader, val_dataloader, optimizer, criterion, log_interval, eval_interval):
    time_start = time.time()
    iter = 0

    if args.model_mode == "18" or args.model_mode == "34" or args.model_mode == 'x50':
        log_file = './logs/resnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode + time.strftime(
            '%m-%d-%H-%M',
            time.localtime(
                time.time())) + '.txt'
        model_path = './models/dual_resnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode
    else:
        log_file = './logs/efficientnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode + time.strftime(
            '%m-%d-%H-%M',
            time.localtime(
                time.time())) + '.txt'
        model_path = './models/dual_efficientnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode

    # if args.loss_type == 'fl':
    #     log_file = './logs/resnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode + args.loss_type + str(
    #         args.gamma) + time.strftime(
    #         '%m-%d-%H-%M',
    #         time.localtime(
    #             time.time())) + '.txt'
    #     model_path = './models/dual_resnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode + args.loss_type + str(
    #         args.gamma)
    # else:
    #     log_file = './logs/resnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode + time.strftime(
    #         '%m-%d-%H-%M',
    #         time.localtime(
    #             time.time())) + '.txt'
    #     model_path = './models/dual_resnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode

    model.train()
    model = model.cuda()

    if os.path.exists(model_path) == False:
        os.makedirs(model_path)

    avg_loss_list = []
    avg_kappa_list = []
    best_kappa = 0.
    best_kappa_100 = 0.
    low_eval_loss = 100
    best_acc1 = 0.
    best_acc2 = 0.
    best_acc12 = 0.
    best_acc = 0.
    best_acc1_100 = 0.
    best_acc2_100 = 0.
    best_acc12_100 = 0.
    best_acc_100 = 0.
    while iter < iters:
        # for data in train_dataloader:
        for fundus_imgs, oct_imgs, labels in train_loader:
            iter += 1
            if iter > iters:
                break
            # fundus_imgs = (data[0] / 255.).astype("float32")
            # oct_imgs = (data[1] / 255.).astype("float32")
            # labels = data[2].astype('int64')

            if args.gaussian == '_gaussian':
                for i in range(fundus_imgs.size(0)):
                    b = random.uniform(0, 1)
                    if (b > 0.6):
                        fundus_imgs1 = np.transpose(fundus_imgs[i], (1, 2, 0))
                        fundus_imgs2 = skimage.util.random_noise(fundus_imgs1, mode='gaussian', seed=None, clip=True,
                                                                 var=random.uniform(0.01, 0.05))
                        fundus_imgs[i] = trans.ToTensor()(fundus_imgs2)

            optimizer.zero_grad()

            fundus_imgs = Variable(fundus_imgs.cuda())  # 同理
            oct_imgs = Variable(oct_imgs.cuda())  # 同理
            labels = Variable(labels.cuda())  # 同理

            logits = model(fundus_imgs, oct_imgs)
            loss = criterion(logits, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            for p, l in zip(logits.detach().cpu().numpy().argmax(1), labels.detach().cpu().numpy()):
                avg_kappa_list.append([p, l])

            loss.backward()
            optimizer.step()

            # model.clear_gradients()
            # pdb.set_trace()
            avg_loss_list.append(loss.detach().cpu().numpy())  ##.detach().cpu().numpy()[0]

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_kappa_list = np.array(avg_kappa_list)

                # pdb.set_trace()
                avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1], weights='quadratic')
                pred = avg_kappa_list[:, 0]
                gt = avg_kappa_list[:, 1]

                try:
                    acc0 = pred[gt == 0].tolist().count(0) / len(pred[gt == 0])
                    acc1 = pred[gt == 1].tolist().count(1) / len(pred[gt == 1])
                    acc2 = pred[gt == 2].tolist().count(2) / len(pred[gt == 2])
                    acc12 = (pred[gt == 1].tolist().count(1) + pred[gt == 2].tolist().count(2)) / (
                            len(pred[gt == 1]) + len(pred[gt == 2]))
                    acc = np.sum(pred == gt) / len(pred)
                except:
                    acc0, acc1, acc2, acc12, acc = 0, 0, 0, 0, 0

                avg_loss_list = []
                avg_kappa_list = []
                f = open(log_file, 'a', encoding='utf-8')
                f.write(
                    "[TRAIN] iter={}/{} avg_loss={:.4f} avg_kappa={:.4f} acc0={:.4f} acc1={:.4f} acc2={:.4f} acc12={:.4f} acc={:.4f}".format(
                        iter, iters, avg_loss, avg_kappa, acc0, acc1, acc2, acc12, acc))
                f.write('\n')
                f.close()
                print(
                    "[TRAIN] iter={}/{} avg_loss={:.4f} avg_kappa={:.4f} acc0={:.4f} acc1={:.4f} acc2={:.4f} acc12={:.4f} acc={:.4f}".format(
                        iter, iters, avg_loss, avg_kappa, acc0, acc1, acc2, acc12, acc))
                print('time cost: {:2.2f}s'.format(time.time() - time_start))
                time_start = time.time()
                avg_kappa_train = avg_kappa

            if iter % eval_interval == 0:
                avg_loss, avg_kappa, acc0, acc1, acc2, acc12, acc = val(model, val_dataloader, criterion)
                f = open(log_file, 'a', encoding='utf-8')
                f.write(
                    "[EVAL] iter={}/{} avg_loss={:.4f} kappa={:.4f} acc0={:.4f} acc1={:.4f} acc2={:.4f} acc12={:.4f} acc={:.4f}".format(
                        iter, iters, avg_loss, avg_kappa, acc0, acc1, acc2, acc12, acc))
                f.write('\n')
                f.close()
                print(
                    "[EVAL] iter={}/{} avg_loss={:.4f} kappa={:.4f} acc0={:.4f} acc1={:.4f} acc2={:.4f} acc12={:.4f} acc={:.4f}".format(
                        iter, iters, avg_loss, avg_kappa, acc0, acc1, acc2, acc12, acc))
                if avg_kappa >= best_kappa_100 and avg_kappa_train == 1 and avg_kappa > 0.75:
                    best_kappa_100 = avg_kappa
                    torch.save(model,
                               os.path.join(model_path,
                                            "best_model100_{:.4f}.pth".format(best_kappa_100)))  ###torch1.6
                elif avg_kappa == best_kappa_100 and avg_kappa_train == 1 and avg_kappa > 0.75:
                    best_kappa_100 = avg_kappa
                    if avg_loss < low_eval_loss:
                        low_eval_loss = avg_loss
                        torch.save(model,
                                   os.path.join(model_path,
                                                "best_model100_{:.4f}.pth".format(best_kappa_100)))  ###torch1.6
                # elif avg_kappa >= best_kappa and avg_kappa > 0.75:
                #     best_kappa = avg_kappa
                #     torch.save(model,
                #                os.path.join(model_path,
                #                             "best_model_{:.4f}.pth".format(best_kappa)))

                if acc12 >= best_acc12_100 and avg_kappa_train == 1 and acc12 > 0.70 and acc0 == 1 and acc2 == 1:
                    best_acc12_100 = acc12
                    torch.save(model,
                               os.path.join(model_path,
                                            "best_model100acc1202_{:.4f}.pth".format(best_acc12_100)))  ###torch1.6
                elif acc12 >= best_acc12_100 and avg_kappa_train == 1 and acc12 > 0.70 and acc2 == 1:
                    best_acc12_100 = acc12
                    torch.save(model,
                               os.path.join(model_path,
                                            "best_model100acc122_{:.4f}.pth".format(best_acc12_100)))  ###torch1.6
                elif acc12 >= best_acc12_100 and avg_kappa_train == 1 and acc12 > 0.70:
                    best_acc12_100 = acc12
                    torch.save(model,
                               os.path.join(model_path,
                                            "best_model100acc12_{:.4f}.pth".format(best_acc12_100)))  ###torch1.6
                # elif acc12 >= best_acc12 and acc12 > 0.70:
                #     best_acc12 = acc12
                #     torch.save(model,
                #                os.path.join(model_path,
                #                             "best_modelacc12_{:.4f}.pth".format(best_acc12)))

            model.train()


def val(model, var_loader, criterion):
    model.eval()
    avg_loss_list = []
    cache = []
    with torch.no_grad():
        # for data in val_dataloader:
        for fundus_imgs, oct_imgs, labels in var_loader:
            # fundus_imgs = (data[0] / 255.).astype("float32")
            # oct_imgs = (data[1] / 255.).astype("float32")
            # labels = data[2].astype('int64')

            fundus_imgs = Variable(fundus_imgs.cuda())  # 同理
            oct_imgs = Variable(oct_imgs.cuda())  # 同理
            labels = Variable(labels.cuda())  # 同理

            logits = model(fundus_imgs, oct_imgs)
            for p, l in zip(logits.detach().cpu().numpy().argmax(1), labels.detach().cpu().numpy()):
                cache.append([p, l])

            loss = criterion(logits, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            # avg_loss_list.append(loss.numpy()[0])
            avg_loss_list.append(loss.detach().cpu().numpy())
    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
    avg_loss = np.array(avg_loss_list).mean()

    pred = cache[:, 0]
    gt = cache[:, 1]
    try:
        acc0 = pred[gt == 0].tolist().count(0) / len(pred[gt == 0])
        acc1 = pred[gt == 1].tolist().count(1) / len(pred[gt == 1])
        acc2 = pred[gt == 2].tolist().count(2) / len(pred[gt == 2])
        acc12 = (pred[gt == 1].tolist().count(1) + pred[gt == 2].tolist().count(2)) / (
                len(pred[gt == 1]) + len(pred[gt == 2]))
        acc = np.sum(pred == gt) / len(pred)
    except:
        acc0, acc1, acc2, acc12, acc = 0, 0, 0, 0, 0

    return avg_loss, kappa, acc0, acc1, acc2, acc12, acc


def get_dataloader(args):
    filelists = os.listdir(args.train_root)
    train_filelists, val_filelists = train_test_split(filelists, test_size=args.val_ratio, random_state=42)

    # train_filelists_v1 = ['0068', '0034', '0084', '0071', '0001', '0100', '0048', '0028', '0035', '0010', '0085',
    #                       '0059', '0088', '0026', '0025', '0086', '0031', '0038', '0041', '0002', '0073', '0094',
    #                       '0096', '0003', '0011', '0019', '0054', '0069', '0007', '0043', '0030', '0060', '0027',
    #                       '0066', '0045', '0093', '0039', '0037', '0087', '0058', '0052', '0067', '0016', '0044',
    #                       '0095', '0015', '0079', '0024', '0042', '0013', '0053', '0091', '0008', '0040', '0005',
    #                       '0099', '0074', '0061', '0051', '0020', '0033', '0081', '0077', '0004', '0006', '0050',
    #                       '0062', '0056', '0032', '0057', '0090', '0046', '0098', '0070', '0022', '0018', '0009',
    #                       '0078', '0029', '0012']
    # val_filelists_v1 = ['0080', '0072', '0023', '0089', '0017', '0075', '0097', '0064', '0021', '0049', '0065', '0047',
    #                     '0036', '0014', '0076', '0055', '0083', '0092', '0063', '0082']  # 9 6 5
    #
    # train_filelists_v2 = ['0073', '0094',
    #                       '0096', '0003', '0011', '0019', '0054', '0069', '0007', '0043', '0030', '0060', '0027',
    #                       '0066', '0045', '0093', '0039', '0037', '0087', '0058', '0052', '0067', '0016', '0044',
    #                       '0095', '0015', '0079', '0024', '0042', '0013', '0053', '0091', '0008', '0040', '0005',
    #                       '0099', '0074', '0061', '0051', '0020', '0033', '0081', '0077', '0004', '0006', '0050',
    #                       '0062', '0056', '0032', '0057', '0090', '0046', '0098', '0070', '0022', '0018', '0009',
    #                       '0078', '0029', '0012', '0080', '0072', '0023', '0089', '0017', '0075', '0097', '0064',
    #                       '0021', '0049', '0065', '0047',
    #                       '0036', '0014', '0076', '0055', '0083', '0092', '0063', '0082']
    # val_filelists_v2 = ['0068', '0034', '0084', '0071', '0001', '0100', '0048', '0028', '0035', '0010', '0085',
    #                     '0059', '0088', '0026', '0025', '0086', '0031', '0038', '0041', '0002']  # 9 6 5
    #
    # train_filelists_v3 = ['0080', '0072', '0023', '0089', '0017', '0075', '0097', '0064', '0021', '0049', '0065',
    #                       '0047',
    #                       '0036', '0014', '0076', '0055', '0083', '0092', '0063', '0082', '0068', '0034', '0084',
    #                       '0071', '0001', '0100', '0048', '0028', '0035', '0010', '0085',
    #                       '0059', '0088', '0026', '0025', '0086', '0031', '0038', '0041', '0002', '0073', '0094',
    #                       '0096', '0003', '0011', '0019', '0054', '0069', '0007', '0043', '0030', '0060', '0027',
    #                       '0066', '0045', '0093', '0039', '0037', '0087', '0058', '0052', '0067', '0016', '0044',
    #                       '0095', '0015', '0079', '0024', '0042', '0013', '0053', '0091', '0008', '0040', '0005',
    #                       '0099', '0074', '0061', '0051', '0020']
    # val_filelists_v3 = ['0033', '0081', '0077', '0004', '0006', '0050',
    #                     '0062', '0056', '0032', '0057', '0090', '0046', '0098', '0070', '0022', '0018', '0009',
    #                     '0078', '0029', '0012']  # 9 5 6  v5->v3

    # train_filelists_v4 = ['0052', '0067', '0016', '0044','0073',
    #                       '0095', '0015', '0079', '0024', '0013', '0053', '0091', '0040', '0005',
    #                       '0099', '0074', '0061', '0051', '0020', '0033', '0081', '0077', '0004', '0006', '0050',
    #                       '0062', '0056', '0032', '0057', '0090', '0046', '0098', '0070', '0022', '0018',
    #                       '0078', '0029', '0012', '0080', '0072', '0023', '0089', '0017', '0075', '0097', '0064',
    #                       '0021', '0049', '0065', '0047', '0003', '0054',
    #                       '0036', '0014', '0076', '0055', '0083', '0092', '0063', '0082', '0068', '0034', '0084',
    #                       '0071', '0001', '0100', '0048', '0028', '0035', '0010', '0085',
    #                       '0059', '0088', '0026', '0025', '0086', '0031', '0038', '0041', '0002']
    # val_filelists_v4 = ['0094', '0008', '0009', '0042',
    #                     '0096', '0011', '0019',  '0069', '0007', '0043', '0030', '0060', '0027',
    #                     '0066', '0045', '0093', '0039', '0037', '0087', '0058']  # 9 5 6

    train_filelists = ['0081', '0077', '0004', '0006', '0050',
                          '0062', '0056', '0032', '0057', '0090', '0046', '0098', '0070', '0022', '0018', '0009',
                          '0078', '0029', '0012', '0080', '0072', '0023', '0089', '0017', '0075', '0097', '0064',
                          '0021', '0049', '0065', '0047','0052', '0099',
                          '0036', '0014', '0076', '0055', '0083', '0092', '0063', '0082', '0068', '0034', '0084',
                          '0071', '0001', '0100', '0048', '0028', '0035', '0010', '0085',
                          '0059', '0088', '0026', '0025', '0086', '0031', '0038', '0041', '0002', '0073',
                          '0096', '0003', '0011', '0019', '0054', '0069', '0007', '0043', '0030', '0060', '0027',
                          '0066', '0045', '0093', '0039', '0037', '0087', '0058']
    val_filelists = ['0067', '0016', '0044', '0094', '0033',
                        '0095', '0015', '0079', '0024', '0042', '0013', '0053', '0091', '0008', '0040', '0005',
                        '0074', '0061', '0051', '0020']  # 9 6 5



    print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))


    # for id in train_filelists:
    #     f = open('trainval_split.txt', 'a', encoding='utf-8')
    #     f.write(id + '\n')
    #     f.close()
    #
    # for id in val_filelists:
    #     f = open('trainval_split.txt', 'a', encoding='utf-8')
    #     f.write(id + '\n')
    #     f.close()

    if args.transforms_mode == 'centerv2':
        img_train_transforms = trans.Compose([
            trans.RandomResizedCrop(
                args.image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip(),
            trans.RandomRotation(30),
            trans.ToTensor()
        ])

    oct_train_transforms = trans.Compose([
        trans.ToTensor(),
        # trans.RandomHorizontalFlip(),
        # trans.RandomVerticalFlip()
    ])

    if args.transforms_mode == 'centerv2':
        img_val_transforms = trans.Compose([
            trans.CenterCrop((2000, 2000)),
            trans.Resize((args.image_size, args.image_size)),
            trans.ToTensor(),
        ])

    oct_val_transforms = trans.Compose([
        # trans.CenterCrop(512),
        trans.ToTensor(),
    ])

    if args.sub1dataset_mode == 'orig':
        train_dataset = GAMMA_sub1_dataset(dataset_root=args.train_root,
                                           img_transforms=img_train_transforms,
                                           oct_transforms=oct_train_transforms,
                                           filelists=train_filelists,
                                           label_file=args.label_file)

        val_dataset = GAMMA_sub1_dataset(dataset_root=args.train_root,
                                         img_transforms=img_val_transforms,
                                         oct_transforms=oct_val_transforms,
                                         filelists=val_filelists,
                                         label_file=args.label_file)
    elif args.sub1dataset_mode == 'crop':
        train_dataset = GAMMA_sub1_dataset_512(dataset_root=args.train_root,
                                               img_transforms=img_train_transforms,
                                               oct_transforms=oct_train_transforms,
                                               filelists=train_filelists,
                                               label_file=args.label_file)

        val_dataset = GAMMA_sub1_dataset_512(dataset_root=args.train_root,
                                             img_transforms=img_val_transforms,
                                             oct_transforms=oct_val_transforms,
                                             filelists=val_filelists,
                                             label_file=args.label_file)
    elif args.sub1dataset_mode == 'crop_cup':
        train_dataset = GAMMA_sub1_dataset_512_cup(dataset_root=args.train_root,
                                                   img_transforms=img_train_transforms,
                                                   oct_transforms=oct_train_transforms,
                                                   filelists=train_filelists,
                                                   label_file=args.label_file)

        val_dataset = GAMMA_sub1_dataset_512_cup(dataset_root=args.train_root,
                                                 img_transforms=img_val_transforms,
                                                 oct_transforms=oct_val_transforms,
                                                 filelists=val_filelists,
                                                 label_file=args.label_file)
    elif args.sub1dataset_mode == 'small':
        train_dataset = GAMMA_sub1_dataset_224(dataset_root=args.train_root,
                                               img_transforms=img_train_transforms,
                                               oct_transforms=oct_train_transforms,
                                               filelists=train_filelists,
                                               label_file=args.label_file)

        val_dataset = GAMMA_sub1_dataset_224(dataset_root=args.train_root,
                                             img_transforms=img_val_transforms,
                                             oct_transforms=oct_val_transforms,
                                             filelists=val_filelists,
                                             label_file=args.label_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False
    )

    return train_loader, val_loader


def main():
    args = get_argparser().parse_args()

    train_loader, val_dataloader = get_dataloader(args)

    if args.model_mode == '18':
        model = Dual_network18()
    elif args.model_mode == '34':
        model = Dual_network34()
    elif args.model_mode == 'x50':
        model = Dual_networkx50()
    elif args.model_mode == 'b0':
        model = Dual_efficientnetb0()
    elif args.model_mode == 'b1':
        model = Dual_efficientnetb1()
    elif args.model_mode == 'b2':
        model = Dual_efficientnetb2()
    elif args.model_mode == 'b3':
        model = Dual_efficientnetb3()
    elif args.model_mode == 'b4':
        model = Dual_efficientnetb4()
    elif args.model_mode == 'b5':
        model = Dual_efficientnetb5()
    elif args.model_mode == 'b6':
        model = Dual_efficientnetb6()
    elif args.model_mode == 'b7':
        model = Dual_efficientnetb7()

    if args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.loss_type == 'fl':
        criterion = FocalLoss(class_num=3, gamma=args.gamma)
    elif args.loss_type == 'ce':
        criterion = torch.nn.CrossEntropyLoss()

    train(args, model, args.iters, train_loader, val_dataloader, optimizer, criterion, log_interval=10,
          eval_interval=10)


if __name__ == '__main__':
    main()