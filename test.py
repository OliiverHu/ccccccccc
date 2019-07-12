import numpy as np
import cv2
from utils.image import apply_random_scale_and_crop, random_distort_image, random_flip, correct_bounding_boxes


def _aug_image(instance, net_h, net_w):
    image_name = instance
    # image = cv2.imread(image_name)  # RGB image
    image = np.zeros([20, 20, 5])
    if image is None: print('Cannot find ', image_name)
    image = image[:, :, ::-1]  # RGB image

    image_h, image_w, _ = image.shape

    # determine the amount of scaling and cropping
    # dw = 0.3 * image_w
    # dh = 0.3 * image_h
    dw = 0
    dh = 0

    new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
    # scale = np.random.uniform(0.25, 2)
    scale = np.random.uniform(0.85, 1.15)

    if new_ar < 1:
        new_h = int(scale * net_h)
        new_w = int(net_h * new_ar)
    else:
        new_w = int(scale * net_w)
        new_h = int(net_w / new_ar)

    dx = int(np.random.uniform(0, net_w - new_w))
    dy = int(np.random.uniform(0, net_h - new_h))

    # apply scaling and cropping
    im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)

    # randomly distort hsv space
    # im_sized = random_distort_image(im_sized)

    # randomly flip
    flip = np.random.randint(2)
    im_sized = random_flip(im_sized, flip)

    # correct the size and pos of bounding boxes

    return im_sized


def _aug_image_ini(instance, net_h, net_w):
    image_name = instance
    image = cv2.imread(image_name)  # RGB image

    if image is None: print('Cannot find ', image_name)
    image = image[:, :, ::-1]  # RGB image

    image_h, image_w, _ = image.shape

    # determine the amount of scaling and cropping
    dw = 0.3 * image_w
    dh = 0.3 * image_h
    # dw = 0
    # dh = 0

    new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
    scale = np.random.uniform(0.25, 2)
    # scale = np.random.uniform(0.85, 1.15)

    if new_ar < 1:
        new_h = int(scale * net_h)
        new_w = int(net_h * new_ar)
    else:
        new_w = int(scale * net_w)
        new_h = int(net_w / new_ar)

    dx = int(np.random.uniform(0, net_w - new_w))
    dy = int(np.random.uniform(0, net_h - new_h))

    # apply scaling and cropping
    im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)

    # randomly distort hsv space
    im_sized = random_distort_image(im_sized)

    # randomly flip
    flip = np.random.randint(2)
    im_sized = random_flip(im_sized, flip)

    # correct the size and pos of bounding boxes

    return im_sized


img_name = "E:/Training/VOCdevkit/VOC2012/JPEGImages/" + "2007_002824.jpg"
img_ini = cv2.imread(img_name)
cv2.imshow('ini', img_ini)
while 1:
    img = _aug_image(img_name, 416, 416)
    cv2.imshow('sized', img)
    img = _aug_image_ini(img_name, 416, 416)
    cv2.imshow('sized_dis', img)
    cv2.waitKey()

# a = np.zeros([2,2,3])
# # a = [[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]]
# count = 0
# for j in range(2):
#     for i in range(2):
#         for k in range(3):
#             a[i][j][k] = count
#             count += 1
# print("a=")
# print(a)
# a = a[:,:,::-1]
# print("a=")
# print(a)