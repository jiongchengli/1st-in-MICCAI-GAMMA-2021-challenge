import os
import cv2


def valid_result(best_result_path, mask_result_path, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for mask_id in range(101, 201):
        mask_name = '%04d.bmp' % mask_id

        best_mask = cv2.imread(os.path.join(best_result_path, mask_name))
        result = best_mask.copy()

        mask = cv2.imread(os.path.join(mask_result_path, mask_name))
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        gray = cv2.Canny(gray, 128, 255)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        binary, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(result, binary, -1, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(save_dir, mask_name), result)


if __name__ == '__main__':
    best_result_path = os.path.join('data', 'refuge', 'Test_result_ensemble_best')
    mask_result_path = os.path.join('data', 'refuge', 'Test_result_ensemble')
    save_dir = os.path.join('data', 'refuge', 'Test_result_valid')
    valid_result(best_result_path, mask_result_path, save_dir)
