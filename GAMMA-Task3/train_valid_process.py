import os
from datasets.data_utils import gen_txt, crop_img_mask


if __name__ == '__main__':
    root = os.path.join('data', 'refuge')
    split2txt = {
        'Train': os.path.join(root, 'train.txt'),
        'Train_mask': os.path.join(root, 'train_mask.txt'),
        'Valid': os.path.join(root, 'valid.txt'),
        'Valid_mask': os.path.join(root, 'valid_mask.txt')
    }
    for k, v in split2txt.items():
        print('===> Converting {} to {}'.format(k, v))
        gen_txt(root, k, v)

    train_dir = 'Train_crop'
    train_txt_path = os.path.join(root, 'train.txt')
    train_mask_txt_path = os.path.join(root, 'train_mask.txt')
    crop_img_mask(root, train_txt_path, train_mask_txt_path, train_dir)

    valid_dir = 'Valid_crop'
    valid_txt_path = os.path.join(root, 'valid.txt')
    valid_mask_txt_path = os.path.join(root, 'valid_mask.txt')
    crop_img_mask(root, valid_txt_path, valid_mask_txt_path, valid_dir)
