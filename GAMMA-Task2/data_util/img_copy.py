import os
import shutil


path = './dataset/Train_data_part2/datas_part2/muti-modality_images'
for d in os.listdir(path):
    if d[0] == '.':
        continue
    img_path = path + '/' + d + '/' + d + '.jpg'
    print(img_path)
    shutil.copy(img_path, './dataset/loc_img')
