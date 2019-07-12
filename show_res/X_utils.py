import numpy as np
import cv2
from scipy.special import expit
import show_res.XYM_show as XYM_show


def get_yolo_boxes(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
    image_h, image_w, _ = images[0].shape
    nb_images = len(images)
    batch_input = np.zeros((nb_images, net_h, net_w, 3))

    # preprocess the input
    for i in range(nb_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)

        # run the prediction
    batch_output = model.predict_on_batch(batch_input)
    batch_boxes = [None] * nb_images

    for i in range(nb_images):
        yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        boxes = []

        # decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2 - j) * 6:(3 - j) * 6]  # config['model']['anchors']
            boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

    return batch_boxes


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1]/255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    # resized = cv2.resize(image[:, :, ::-1], (new_w, new_h))
    #
    # # embed the image into the standard letter box
    # new_image = np.ones((net_h, net_w, 3)) * 127
    # new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
    # new_image = np.expand_dims(new_image, 0)

    return new_image


def decode_netout(netout_ini, img_name):
    netout = netout_ini.copy()
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    img_path = 'E:/Training/VOCdevkit/VOC2012/my_training/Class5_picture_out/'

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4] = _sigmoid(netout[..., 4])
    # netout[..., 5:] *= netout[..., 5:] > obj_thresh
    test_class_p = netout[..., 5:]

    img = netout[..., 4].copy()
    img = np.amax(img, -1)
    img = XYM_show.data2img(img, grid_h, grid_w)
    img = XYM_show.exp_img(img, 416 // grid_h)
    cv2.imwrite(img_path + img_name + '_box.jpg', img)

    img = netout[..., 5:].copy()
    img = np.amax(img, -1)
    # img = img[..., 4]
    img = np.amax(img, -1)
    img = XYM_show.data2img(img, grid_h, grid_w)
    img = XYM_show.exp_img(img, 416 // grid_h)
    cv2.imwrite(img_path + img_name + '_class.jpg', img)

    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    img = netout[..., 5:].copy()
    img = np.amax(img, -1)
    # img = img[..., 4]
    img = np.amax(img, -1)
    img = XYM_show.data2img(img, grid_h, grid_w)
    img = XYM_show.exp_img(img, 416 // grid_h)
    cv2.imwrite(img_path + img_name + '_class_box.jpg', img)

    return


def _sigmoid(x):
    return expit(x)


def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)