import os
import cv2
import numpy as np
from PIL import Image


def gen_txt(root, split, txt_file):
    path = os.path.join(root, split)
    if not os.path.exists(path):
        os.mkdir(path)

    img_list = os.listdir(path)
    img_list.sort()

    lines = []
    for i, name in enumerate(img_list):
        lines.append(root + '/' + split + '/' + name + '\n')

    with open(txt_file, 'w') as f:
        f.writelines(lines)


def crop_img(root, img_txt_path, save_dir, f_crop):
    imgs = []
    img_names = []

    f_img = open(img_txt_path, 'r')
    for img in f_img:
        img = img.rstrip()
        imgs.append(img)
        img_names.append(img.split('/')[-1])
    f_img.close()

    radius = 21
    crop_info = ''
    f_crop = open(os.path.join(root, f_crop), 'w')

    center_w = 1900
    center_h = 1200

    for i in range(len(imgs)):
        print(i + 1)

        x_path = imgs[i]
        x_name = img_names[i].replace('jpg', 'png')
        img_x = Image.open(x_path).convert('RGB')

        img_w, img_h = img_x.size
        x_center = img_w // 2
        y_center = img_h // 2

        # 读取图片并将其转化为灰度图片
        image = cv2.imread(x_path)
        image_crop = image.copy()
        image_crop = image_crop[y_center - center_h // 2:y_center + center_h // 2,
                     x_center - center_w // 2:x_center + center_w // 2]
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)

        # 应用高斯模糊进行预处理
        gray = cv2.GaussianBlur(gray, (radius, radius), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

        c_dim0 = maxLoc[0]
        c_dim1 = maxLoc[1]

        x_loc = c_dim0 + ((img_w - center_w) // 2)
        y_loc = c_dim1 + ((img_h - center_h) // 2)

        crop_info += '{0} {1} {2} {3} {4}\n'.format(x_name, str(img_w), str(img_h),
                                                    x_loc, y_loc)

        center_dim0, center_dim1 = x_loc, y_loc

        # crop size (512, 512)
        if center_dim0 >= 256 and center_dim1 >= 256:
            img_x = img_x.crop((center_dim0 - 256, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
        elif center_dim0 < 256 and center_dim1 >= 256:
            img_x_crop = img_x.crop((0, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_black_rgb.paste(img_x_crop, (256 - center_dim0, 0, 512, 512))
            img_x = img_black_rgb
        elif center_dim0 >= 256 and center_dim1 < 256:
            img_x_crop = img_x.crop((center_dim0 - 256, 0, center_dim0 + 256, center_dim1 + 256))
            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_black_rgb.paste(img_x_crop, (0, 256 - center_dim1, 512, 512))
            img_x = img_black_rgb
        else:
            img_x_crop = img_x.crop((0, 0, center_dim0 + 256, center_dim1 + 256))
            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_black_rgb.paste(img_x_crop, (256 - center_dim0, 256 - center_dim1, 512, 512))
            img_x = img_black_rgb

        save_path = os.path.join(root, save_dir)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        img_x.convert('RGB').save(os.path.join(save_path, x_name))
    f_crop.write(crop_info)
    f_crop.close()


def crop_img_txt(root, img_txt_path, save_dir, f_crop):
    imgs = []
    img_names = []

    f_img = open(img_txt_path, 'r')
    for img in f_img:
        img = img.rstrip()
        imgs.append(img)
        img_names.append(img.split('/')[-1])
    f_img.close()

    img_dict = {}
    f_crop = open(os.path.join(root, f_crop), 'r')
    for line in f_crop:
        info = line.split(' ')
        key = info[0]
        img_dict[key] = [int(info[3].split('.')[0]), int(info[4].split('.')[0])]
    f_crop.close()

    for i in range(len(imgs)):
        print(i + 1)

        x_path = imgs[i]
        x_name = img_names[i]
        img_x = Image.open(x_path).convert('RGB')

        center_dim0 = img_dict[x_name][0]
        center_dim1 = img_dict[x_name][1]

        # crop size (512, 512)
        if center_dim0 >= 256 and center_dim1 >= 256:
            img_x = img_x.crop((center_dim0 - 256, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
        elif center_dim0 < 256 and center_dim1 >= 256:
            img_x_crop = img_x.crop((0, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_black_rgb.paste(img_x_crop, (256 - center_dim0, 0, 512, 512))
            img_x = img_black_rgb
        elif center_dim0 >= 256 and center_dim1 < 256:
            img_x_crop = img_x.crop((center_dim0 - 256, 0, center_dim0 + 256, center_dim1 + 256))
            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_black_rgb.paste(img_x_crop, (0, 256 - center_dim1, 512, 512))
            img_x = img_black_rgb
        else:
            img_x_crop = img_x.crop((0, 0, center_dim0 + 256, center_dim1 + 256))
            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_black_rgb.paste(img_x_crop, (256 - center_dim0, 256 - center_dim1, 512, 512))
            img_x = img_black_rgb

        save_path = os.path.join(root, save_dir)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        img_x.convert('RGB').save(os.path.join(save_path, x_name.replace('jpg', 'png')))


def crop_img_mask(root, img_txt_path, mask_txt_path, save_dir):
    imgs = []
    img_names = []

    f_img = open(img_txt_path, 'r')
    for img in f_img:
        img = img.rstrip()
        imgs.append([img])
        img_names.append([img.split('/')[-1]])
    f_img.close()

    i = 0
    f_mask = open(mask_txt_path, 'r')
    for mask in f_mask:
        mask = mask.rstrip()
        imgs[i].append(mask)
        img_names[i].append(mask.split('/')[-1])
        i += 1
    f_mask.close()

    for i in range(len(imgs)):
        print(i + 1)

        x_path, y_path = imgs[i]
        x_name, y_name = img_names[i]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path)

        img_y = np.array(img_y)

        loc = np.where(img_y == 128)
        center_dim0 = (np.max(loc[1]) + np.min(loc[1])) // 2
        center_dim1 = (np.max(loc[0]) + np.min(loc[0])) // 2
        img_y = Image.fromarray(img_y)

        # crop size (512, 512)
        if center_dim0 >= 256 and center_dim1 >= 256:
            img_x = img_x.crop((center_dim0 - 256, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
            img_y = img_y.crop((center_dim0 - 256, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
        elif center_dim0 < 256 and center_dim1 >= 256:
            img_x_crop = img_x.crop((0, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
            img_y_crop = img_x.crop((0, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_white_gray = Image.new('L', (512, 512), 255)

            img_black_rgb.paste(img_x_crop, (256 - center_dim0, 0, 512, 512))
            img_x = img_black_rgb
            img_white_gray.paste(img_y_crop, (256 - center_dim0, 0, 512, 512))
            img_y = img_white_gray
        elif center_dim0 >= 256 and center_dim1 < 256:
            img_x_crop = img_x.crop((center_dim0 - 256, 0, center_dim0 + 256, center_dim1 + 256))
            img_y_crop = img_y.crop((center_dim0 - 256, 0, center_dim0 + 256, center_dim1 + 256))
            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_white_gray = Image.new('L', (512, 512), 255)

            img_black_rgb.paste(img_x_crop, (0, 256 - center_dim1, 512, 512))
            img_x = img_black_rgb
            img_white_gray.paste(img_y_crop, (0, 256 - center_dim1, 512, 512))
            img_y = img_white_gray
        else:
            img_x_crop = img_x.crop((0, 0, center_dim0 + 256, center_dim1 + 256))
            img_y_crop = img_y.crop((0, 0, center_dim0 + 256, center_dim1 + 256))

            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_white_gray = Image.new('L', (512, 512), 255)

            img_black_rgb.paste(img_x_crop, (256 - center_dim0, 256 - center_dim1, 512, 512))
            img_x = img_black_rgb
            img_white_gray.paste(img_y_crop, (256 - center_dim0, 256 - center_dim1, 512, 512))
            img_y = img_white_gray

        save_path = os.path.join(root, save_dir)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        img_x.convert('RGB').save(os.path.join(save_path, x_name.split('.')[0] + '.png'))
        img_y.convert('L').save(os.path.join(save_path, y_name.split('.')[0] + '_mask.png'))


def crop_img_mask_test(root, img_txt_path, mask_txt_path, f_crop, save_dir):
    imgs = []
    img_names = []

    f_img = open(img_txt_path, 'r')
    for img in f_img:
        img = img.rstrip()
        imgs.append([img])
        img_names.append([img.split('/')[-1]])
    f_img.close()

    i = 0
    f_mask = open(mask_txt_path, 'r')
    for mask in f_mask:
        mask = mask.rstrip()
        imgs[i].append(mask)
        img_names[i].append(mask.split('/')[-1])
        i += 1
    f_mask.close()

    img_dict = {}
    f_crop = open(os.path.join(root, f_crop), 'r')
    for line in f_crop:
        info = line.split(' ')
        key = info[0]
        img_dict[key] = [int(info[3]), int(info[4])]
    f_crop.close()

    # loc_info = ''
    # f_loc = open(os.path.join(root, 'loc_info.txt'), 'w')

    for i in range(len(imgs)):
        print(i + 1)

        x_path, y_path = imgs[i]
        x_name, y_name = img_names[i]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path)

        center_dim0 = img_dict[y_name][0]
        center_dim1 = img_dict[y_name][1]

        # img_w, img_h = img_x.size
        # x_center = img_w // 2

        # if abs(center_dim0 - x_center) <= 200:
        #     loc_info += '{0} {1}\n'.format(x_name, 1)
        # else:
        #     loc_info += '{0} {1}\n'.format(x_name, 0)

        # crop size (512, 512)
        if center_dim0 >= 256 and center_dim1 >= 256:
            img_x = img_x.crop((center_dim0 - 256, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
            img_y = img_y.crop((center_dim0 - 256, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
        elif center_dim0 < 256 and center_dim1 >= 256:
            img_x_crop = img_x.crop((0, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
            img_y_crop = img_x.crop((0, center_dim1 - 256, center_dim0 + 256, center_dim1 + 256))
            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_white_gray = Image.new('L', (512, 512), 255)

            img_black_rgb.paste(img_x_crop, (256 - center_dim0, 0, 512, 512))
            img_x = img_black_rgb
            img_white_gray.paste(img_y_crop, (256 - center_dim0, 0, 512, 512))
            img_y = img_white_gray
        elif center_dim0 >= 256 and center_dim1 < 256:
            img_x_crop = img_x.crop((center_dim0 - 256, 0, center_dim0 + 256, center_dim1 + 256))
            img_y_crop = img_y.crop((center_dim0 - 256, 0, center_dim0 + 256, center_dim1 + 256))
            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_white_gray = Image.new('L', (512, 512), 255)

            img_black_rgb.paste(img_x_crop, (0, 256 - center_dim1, 512, 512))
            img_x = img_black_rgb
            img_white_gray.paste(img_y_crop, (0, 256 - center_dim1, 512, 512))
            img_y = img_white_gray
        else:
            img_x_crop = img_x.crop((0, 0, center_dim0 + 256, center_dim1 + 256))
            img_y_crop = img_y.crop((0, 0, center_dim0 + 256, center_dim1 + 256))

            img_black_rgb = Image.new('RGB', (512, 512), (0, 0, 0))
            img_white_gray = Image.new('L', (512, 512), 255)

            img_black_rgb.paste(img_x_crop, (256 - center_dim0, 256 - center_dim1, 512, 512))
            img_x = img_black_rgb
            img_white_gray.paste(img_y_crop, (256 - center_dim0, 256 - center_dim1, 512, 512))
            img_y = img_white_gray

        save_path = os.path.join(root, save_dir)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        img_x.convert('RGB').save(os.path.join(save_path, x_name.split('.')[0] + '.png'))
        img_y.convert('L').save(os.path.join(save_path, y_name.split('.')[0] + '_mask.png'))

    # f_loc.write(loc_info)
    # f_loc.close()


def convert_img(opts, save_dir):
    img_w = opts.crop_size
    img_h = opts.crop_size

    img_list = os.listdir(save_dir)
    img_list.sort()

    for img_name in img_list:
        print(img_name)
        img = cv2.imread(os.path.join(save_dir, img_name))
        for i in range(img_w):
            for j in range(img_h):
                b, g, r = img[i, j]
                if b == 128 and g == 0 and r == 0:  # 蓝色
                    img[i, j] = (128, 128, 128)
                elif b == 0 and g == 128 and r == 0:  # 绿色
                    img[i, j] = (0, 0, 0)
                else:  # 黑色
                    img[i, j] = (255, 255, 255)

        cv2.imwrite("./{0}/{1}".format(save_dir, img_name), img)


def recover_img(f_crop_path, pre_mask_dir, mask_dir):
    img_info = []
    f_crop = open(f_crop_path, 'r')
    for img in f_crop:
        img = img.rstrip().split(' ')
        img_info.append(img)
    f_crop.close()

    crop_size = 512
    radius = crop_size // 2

    for i in range(len(img_info)):
        print(i + 1)

        center_x = int(img_info[i][3].split('.')[0])
        center_y = int(img_info[i][4].split('.')[0])

        if 'Disc_Cup_Segmentations' in mask_dir:
            mask_name = img_info[i][0].replace('png', 'bmp')
        else:
            mask_name = img_info[i][0]

        img_mask = Image.open(os.path.join(pre_mask_dir, mask_name))
        img_w = int(img_info[i][1])
        img_h = int(img_info[i][2])
        img_white_gray = Image.new('L', (img_w, img_h), 255)

        img_white_gray.paste(img_mask, (center_x - radius, center_y - radius, center_x + radius, center_y + radius))

        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)

        img_name = os.path.join(mask_dir, mask_name)
        img_white_gray.convert('L').save(img_name)


def get_center(f_crop_path, mask_dir):
    crop_info = ''
    f_crop = open(f_crop_path, 'w')

    img_list = os.listdir(mask_dir)
    img_list.sort()

    i = 1
    for img_name in img_list:
        print(i)
        mask_path = os.path.join(mask_dir, img_name)
        mask_w, mask_h, contours_info = post_process(mask_path)
        x_loc = contours_info[0][2]
        y_loc = contours_info[0][3]
        crop_info += "{0} {1} {2} {3} {4}\n".format(img_name, mask_w, mask_h, x_loc, y_loc)
        i += 1

    f_crop.write(crop_info)
    f_crop.close()


def post_process(mask_path):
    mask = cv2.imread(mask_path)
    mask_h, mask_w = mask.shape[:-1]

    # convert to HSV
    mask_HSV = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # threshold on color
    lower = (0, 0, 0)
    upper = (128, 128, 128)
    thresh = cv2.inRange(mask_HSV, lower, upper)

    # get contours
    result = mask.copy()
    contours_info = []
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    index = 0
    for contr in contours:
        area = cv2.contourArea(contr)
        M = cv2.moments(contr)
        if M["m00"] == 0:
            index += 1
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        contours_info.append((index, area, cx, cy))
        index += 1

    # sort contours by area
    def area_cmp(elem):
        return elem[1]

    # 提取最大轮廓并保存
    contours_info.sort(key=area_cmp, reverse=True)
    if len(contours_info) > 0:
        max_id = contours_info[0][0]
        for idx in range(len(contours_info)):
            if idx == max_id:
                continue
            cv2.fillConvexPoly(result, contours[idx], (255, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=3)
        cv2.imwrite(mask_path, result)
    return mask_w, mask_h, contours_info


def process_img(mask_dir):
    img_list = os.listdir(mask_dir)
    img_list.sort()

    i = 1
    for img_name in img_list:
        print(i)
        mask_path = os.path.join(mask_dir, img_name)
        post_process(mask_path)
        i += 1
