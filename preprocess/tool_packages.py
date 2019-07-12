import csv
import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
from preprocess.coordinates_translator import get_8_point


def get_mhd_path(mhd_dir):
    """
    # mhd_dir: mhd_dir path to a dir containing mhd files
    # return: a list of def path of all mhd files in mhd_dir
    #
    """
    mhd_paths = []

    if os.path.isdir(mhd_dir):
        for inp_file in os.listdir(mhd_dir):
            mhd_paths += [mhd_dir + inp_file]
    else:
        mhd_paths += [mhd_dir]

    mhd_paths = [inp_file for inp_file in mhd_paths if (inp_file[-4:] in ['.mhd'])]

    return mhd_paths


def get_label_coords(annotation_csv, name):
    """
    # to get the label info in csv file
    # annotation_csv: a csv file handler(list)
    # name: the id of mhd file
    # return: all rowdata in annotation file which matches the name(input)
    """
    labels = []  # np.zeros((50, 8), dtype=float)
    for row in annotation_csv:
        if row[0] == name:
            labels.append(row)
        else:
            pass

    return labels


def get_filename(file_path):
    """
    # get file name
    # filepath: the path to mhd(or raw) file
    """
    name = file_path.split('/')[-1].split('.')[0]
    return name


def read_csv(filename):
    """
    # csv file reader
    # filename: the path to annotations 'chestCT_round1_annotation.csv'
    # return: file handler
    """
    lines = []
    with open(filename, "rt") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


def raw_image_reader(filename):
    """
    # itk image loader
    # filename: the path to mhd file
    """
    itkimage = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(itkimage.GetOrigin()))  # CT原点坐标
    spacing = np.array(list(itkimage.GetSpacing()))  # CT像素间隔
    return img, origin, spacing


def raw_image_writer(numpy_array, save_dir):
    writer = sitk.GetImageFromArray(numpy_array)
    sitk.WriteImage(writer, save_dir)


# if __name__ == '__main__':
#     """
#     a simple visualization script to check the labels and windowing, etc...
#     """
#     image_paths = []
#     input_path = 'chestCT_round1/test/'
#     if os.path.isdir(input_path):
#         for inp_file in os.listdir(input_path):
#             image_paths += [input_path + inp_file]
#     else:
#         image_paths += [input_path]
#
#     mhd_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.mhd'])]
#     # 加载结节标注
#     anno_path = 'chestCT_round1_annotation.csv'
#     # temp_csv = 'temp.csv'
#     annos = read_csv(anno_path)
#     progress = 0
#     for mhd_path in mhd_paths:
#         file_name = get_filename(mhd_path)
#         # print(len(annos)) length of the csv file is 12219
#         label = get_label_coords(annos, file_name)
#         print(label)
#         progress += 1
#         print('Progress:' + str(progress) + '/' + str(len(mhd_paths)))
#         numpyImage, numpyOrigin, numpySpacing = raw_image_reader(mhd_path)
#         s, w, h = numpyImage.shape  # (slice,w,h)
#         if len(label) == 0:
#             for i in range(s):
#                     image = np.squeeze(numpyImage[i, ...])
#                     plt.imshow(image, cmap='gray')
#                     plt.axis('on')
#                     plt.title(file_name + '_slice' + str(i), fontsize='large', fontweight='bold')
#                     plt.show()
#                     # plt.savefig(file_name + '_slice' + str(i))
#         else:
#             # with open(temp_csv, 'w', newline='') as csvfile:
#             # writer = csv.writer(csvfile)
#             label_db = []
#             for l in label:
#                 worldCoord = np.asarray([float(l[1]), float(l[2]), float(l[3])])
#                 diameter = np.asarray([float(l[4]), float(l[5]), float(l[6])])
#
#                 maxCoord, _, minCoord = get_8_point(worldCoord, diameter, numpyOrigin, numpySpacing, 1)
#                 if maxCoord[2] == minCoord[2]:
#                     label_db.append([minCoord[0], minCoord[1], maxCoord[0], maxCoord[1], minCoord[2], l[7]])
#                     # writer.writerow('\t')
#                 else:
#                     for i in range(minCoord[2], maxCoord[2]+1, 1):
#                         label_db.append([minCoord[0], minCoord[1], maxCoord[0], maxCoord[1], i, l[7]])
#                     # writer.writerow('\t')
#
#             bbox_label = label_db
#             for i in range(s):
#                 image = np.squeeze(numpyImage[i, ...])  # if the image is 3d, the slice is integer
#                 for label in bbox_label:
#                     '''
#                     bbox drawing with matplotlib
#                     '''
#                     # print(label[4])
#                     # print(i)
#                     if i+1 == int(label[4]):
#                         print('z')
#                         plt.imshow(image, cmap='gray')
#                         plt.gca().add_patch(plt.Rectangle(xy=(int(label[0]), int(label[1])), width=int(label[2]) - int(label[0]),
#                                                           height=int(label[3]) - int(label[1]), edgecolor='#FF0000',
#                                                           fill=False, linewidth=0.5))
#
#                         plt.text(int(label[0]), int(label[1]) - 10, str(int(label[5])), size=10, family="fantasy", color="r",
#                                  style="italic", weight="light")
#                     else:
#                         plt.imshow(image, cmap='gray')
#                 plt.axis('on')
#                 plt.title(file_name + '_slice' + str(i), fontsize='large', fontweight='bold')
#                 plt.show()
#                 # plt.savefig(file_name + '_slice' + str(i))
#                 [p.remove() for p in reversed(plt.gca().patches)]
#                 [p.remove() for p in reversed(plt.gca().texts)]
