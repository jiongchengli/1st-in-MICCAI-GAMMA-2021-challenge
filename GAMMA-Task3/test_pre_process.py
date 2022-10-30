import os
from datasets.data_utils import gen_txt, crop_img, crop_img_txt, process_img, recover_img, get_center, crop_img_mask_test

if __name__ == '__main__':
    root = './data/refuge'
    img_txt_name = 'Test'
    img_txt_path = os.path.join(root, 'test.txt')
    gen_txt(root, img_txt_name, img_txt_path)

    # Test_pre_crop | Test_crop | f_crop.txt | seg_loc.txt
    save_dir = 'Test_crop'
    f_crop = 'f_crop.txt'
    if 'pre' in save_dir:
        if 'f_crop' in f_crop:
            crop_img(root, img_txt_path, save_dir, f_crop)
        else:
            crop_img_txt(root, img_txt_path, save_dir, f_crop)
    else:
        # process mask
        f_crop_path = os.path.join(root, f_crop)
        pre_mask_dir = os.path.join(root, 'Test_pre_mask')
        process_img(pre_mask_dir)

        # recover mask
        mask_dir = os.path.join(root, 'Test_mask')
        recover_img(f_crop_path, pre_mask_dir, mask_dir)

        # get mask txt
        mask_txt_name = 'Test_mask'
        mask_txt_path = os.path.join(root, 'test_mask.txt')
        gen_txt(root, mask_txt_name, mask_txt_path)

        # get disc_cup center
        get_center(f_crop_path, mask_dir)

        # re-crop img-mask
        crop_img_mask_test(root, img_txt_path, mask_txt_path, f_crop, save_dir)


