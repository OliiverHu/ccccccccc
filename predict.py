#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes, get_box_info
from keras.models import load_model
from tqdm import tqdm
import numpy as np
from train import create_model, create_training_instances
from file_reader.raw_reader import raw_reader
from preprocess.img_seg import img_windowing


def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output
    predict_path = args.predict
    if_show      = args.show

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)
    makedirs(predict_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 512, 512 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45#0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])
    # infer_model = load_model('backend')
    print('load model')

    ###############################
    #   Predict bounding boxes 
    ###############################

 # do detection on an image or a set of images
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.mhd'])]

    # the main loop
    for image_path in image_paths:
        print(image_path)
        slice_i = 1
        while slice_i < 1000:
            slice_i += 1
            print('slice:' + str(slice_i))
            image = raw_reader(image_path, slice_i)

            if image is None:
                break

            if if_show:
                image_ini = image[..., 2]
                max_pix = np.max(image_ini)
                min_pix = np.min(image_ini)
                # print(max_pix, min_pix)
                image_ini, _ = img_windowing(image_ini, max_pix, min_pix)
                # cv2.imshow('image_ini', image_ini)
                # cv2.waitKey()
                # image_ini = np.uint8(np.float64((image_ini + 1000) / 1800) * 255)

            (imagename, extension) = os.path.splitext(image_path.split('/')[-1])

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'],
                                   obj_thresh, nms_thresh, imagename)[0]

            line = ''

            textname = predict_path + imagename + '_' + str(slice_i) + '.txt'

            if if_show and len(boxes) > 0:
                # draw bounding boxes on the image using labels
                # print('boxes:' + str(len(boxes)))

                draw_boxes(image_ini, boxes, sorted(config['model']['labels']),
                           obj_thresh, output_path + imagename + '_' + str(slice_i) + '.jpg')
                
                # write the image with bounding boxes to file
                # cv2.imwrite(output_path + imagename + '_' + str(slice_i) + '.jpg', np.uint8(image_ini))

                newline = get_box_info(line, boxes, sorted(config['model']['labels']), obj_thresh)
                with open(textname, 'w') as f:
                    f.write(newline)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', default='config_neu.json',
                           help='path to configuration file')
    argparser.add_argument('-i', '--input', default='E:/Training/chestCT/test_input/',
                           help='path to an image, a directory of images, a video, or webcam')
    argparser.add_argument('-o', '--output', default='E:/Training/chestCT/test_output/',
                           help='path to output directory')
    argparser.add_argument('-p', '--predict', default='E:/Training/chestCT/test_output/',
                           help='path to predict directory')
    argparser.add_argument('-s', '--show', default=True,
                           help='whether to show the result')
    
    args = argparser.parse_args()
    _main_(args)
