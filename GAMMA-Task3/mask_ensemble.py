import os
import numpy as np
import cv2


def mask_ensemble(result_path, ensemble_num, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for mask_id in range(101, 201):
        mask_name = '%04d.bmp' % mask_id
        new_mask = np.ones([512, 512, 3], dtype=np.uint8) * 255

        for i in range(1, ensemble_num + 1):
            # 视盘集成
            if i != 3:
                continue
            mask_path = result_path + str(i)
            mask = cv2.imread(os.path.join(mask_path, mask_name))
            new_mask[np.where(np.logical_or(mask == 128, mask == 0))] = 128

        for i in range(1, ensemble_num + 1):
            # 视杯集成
            if i == 3:
                continue
            mask_path = result_path + str(i)
            mask = cv2.imread(os.path.join(mask_path, mask_name))
            new_mask[mask == 0] = 0

        cv2.imwrite(os.path.join(save_dir, mask_name), new_mask)


if __name__ == '__main__':
    ensemble_num = 4
    result_path = os.path.join('data', 'refuge', 'Test_result')
    save_dir = os.path.join('data', 'refuge', 'Test_result_ensemble')
    mask_ensemble(result_path, ensemble_num, save_dir)
