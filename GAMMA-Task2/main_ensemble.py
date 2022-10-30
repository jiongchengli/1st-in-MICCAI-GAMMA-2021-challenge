import csv
import os

import cv2
import pandas as pd

loc_map = dict()
res_list = ['res-1.csv', 'res-4.csv']
alpha_list = [0.5, 0.5]
vis_path = './vis_result/val_predict_fine_ensemble/'
img_path = './dataset/val_loc/img'
csv_name = './Localization_Results.csv'
n = len(res_list)

if not os.path.exists(vis_path):
    os.mkdir(vis_path)

idx = 0
for res in res_list:
    f_csv = csv.reader(open(res, 'r'))
    for i, row in enumerate(f_csv):
        if i == 0:
            continue
        img = row[0] + '.jpg'
        x, y = row[1], row[2]
        if img not in loc_map:
            loc_map[img] = [0.0, 0.0]

        loc_map[img][0] += alpha_list[idx] * float(x)
        loc_map[img][1] += alpha_list[idx] * float(y)
    idx += 1

cache = []
for k, v in loc_map.items():
    # loc_map[k][0] = v[0] / n
    # loc_map[k][1] = v[1] / n
    print(k, loc_map[k][0], loc_map[k][1])
    cache.append([k[:-4], loc_map[k][0], loc_map[k][1]])
submission_result = pd.DataFrame(cache, columns=['data', 'Fovea_X', 'Fovea_Y'])
submission_result[['data', 'Fovea_X', 'Fovea_Y']].to_csv(csv_name, index=False)

for img_name in os.listdir(img_path):
    im = cv2.imread(img_path + '/' + img_name)
    x, y = int(loc_map[img_name][0]), int(loc_map[img_name][1])
    cv2.line(im, (x, y - 50), (x, y + 50), (0, 0, 255), 2)
    cv2.line(im, (x - 50, y), (x + 50, y), (0, 0, 255), 2)
    im = im[y - 256:y + 256, x - 256:x + 256]
    print(img_name)
    cv2.imwrite(vis_path + img_name, im)
