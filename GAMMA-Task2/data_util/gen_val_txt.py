import os
from PIL import Image


val_txt_path = '../dataset/val_loc_fine/val_name_loc.txt'
val_img_path = '../dataset/val_loc_fine/img'

f = open(val_txt_path, "w")
img_list = os.listdir(val_img_path)
img_list = sorted(img_list)
for img in img_list:
    im = Image.open(val_img_path + '/' + img)
    f.write("{0} {1} {2} {3} {4}\n".format(img, im.size[0], im.size[1], -1, -1))


