import os
import SimpleITK as sitk
import numpy as np
import cv2
import csv
from preprocess.coordinates_translator import get_8_point


def get_label_coords(csv_file, name):  # to get the label info in csv file
    labels = [] # np.zeros((50, 8), dtype=float)
    for row in csv_file:
        if row[0] == name:
            labels.append(row)
        else:
            pass

    return labels


def read_csv(filename):  # csv file reader
    lines = []
    with open(filename, "rt") as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            lines.append(line)
    return lines


def show_img():
    image_paths = []
    input_path = 'chestCT_round1/test/'
    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    mhd_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.mhd'])]

    anno_path = 'chestCT_round1_annotation.csv'
    annos = read_csv(anno_path)

    # mhd_paths = ['E:/Training/chestCT/train_part1/323490.mhd']
    for mhd_path in mhd_paths:
        data = sitk.ReadImage(mhd_path)

        name = mhd_path.split('/')[-1].split('.')[0]
        labels = get_label_coords(annos, name)

        origin = np.array(data.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        print(origin)
        spacing = np.array(data.GetSpacing())  # spacing of voxels in world coor. (mm)
        print(spacing)
        # print(data)
        image = sitk.GetArrayFromImage(data)
        k = 0
        x = image.shape
        print(mhd_path)
        print(x)
        while 1:
            try:
                image_show = np.squeeze(image[k, ...])  # if the image is 3d, the slice is integer
            except IndexError:
                break

            image_show = np.uint8((image_show + 1500) / 3000 * 255)

            for label in labels:
                point0 = np.asarray([float(label[1]), float(label[2]), float(label[3])])
                whl0 = np.asarray([float(label[4]), float(label[5]), float(label[6])])
                pmax, point, pmin = get_8_point(point0, whl0, origin, spacing)
                xmax, ymax, zmax = pmax[0], pmax[1], pmax[2]
                xmin, ymin, zmin = pmin[0], pmin[1], pmin[2]
                if zmin <= k <= zmax:
                    cv2.rectangle(image_show, (xmax, ymax), (xmin, ymin), color=(0, 255, 0))

            # image_show = np.uint16((image_show+1500) * 32768//1500)
            cv2.imshow('show', image_show)
            cv2.waitKey()
            k = k + 1


show_img()
