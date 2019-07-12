import cv2
import numpy as np


def data2img(data_ini, size_h, size_w, color=2):
    data = data_ini.copy()
    data = cv2.resize(data, (size_h, size_w))
    max_data = np.max(data)
    min_data = np.min(data)
    print('max:' + str(max_data))
    print('min:' + str(min_data))

    if max_data < (min_data+1e-5):
        multi = 0
        print('zero layer')
        print(max_data)
        print(min_data)
    else:
        multi = 255/(max_data-min_data)
    img = np.ones((size_h, size_w, 3), np.uint8)
    for h in range(size_h):
        for w in range(size_w):
            img[h][w][color] = int((data[h][w] + min_data) * multi)
    return img


def exp_img(ini_img, multi):
    ini_h, ini_w, _ = ini_img.shape
    try:
        out_img = np.ones((ini_h*multi, ini_w*multi, 3), np.uint8)
    except:
        return None
    for h in range(ini_h*multi):
        for w in range(ini_w*multi):
            out_img[h][w] = ini_img[h//multi][w//multi]
    return out_img


# def composite_img(imgs):
#