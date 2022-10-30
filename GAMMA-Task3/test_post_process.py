import os
from datasets.data_utils import recover_img, process_img

if __name__ == '__main__':
    root = './data/refuge'
    f_crop_path = os.path.join(root, 'f_crop.txt')

    # post process disc_cup
    pre_mask_dir = os.path.join(root, 'Test_result')
    process_img(pre_mask_dir)

    # recover disc_cup
    mask_dir = 'Disc_Cup_Segmentations'
    recover_img(f_crop_path, pre_mask_dir, mask_dir)
