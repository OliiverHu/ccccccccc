import cv2
import numpy as np
import preprocess.tool_packages as tool_packages
from matplotlib import pyplot as plt
from skimage.measure import label
import random
import time


def binary_img_reverse(binary_img):
    """
    :param binary_img:  binary image
    :return: a reversed binary image
    """
    result_ = np.array(binary_img)
    # h, w = binary_img.shape
    # for i in range(w):
    #     for j in range(h):
    #         binary_img[i][j] = 1 - binary_img[i][j]

    return ~result_


def largest_connect_area(binary_img):
    """
    Utility: return the largest Connect area of a labeled image
    Parameters: binary_img: binary image
    """

    labeled_img, num = label(binary_img, neighbors=4, background=0, return_num=True)
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    for i in range(1, num):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lca = (labeled_img == max_label)
    # print(lca)
    return lca


def random_sampling(dir_path):
    """
    :param dir_path: path to mhd file directory
    :return: randomly selected samples from the directory pool
    """
    mhd_path_list = tool_packages.get_path(dir_path, 'mhd')
    # sample_count = int(len(mhd_path_list) / 20)
    sample_count = 1
    sampling = random.sample(mhd_path_list, sample_count)
    return sampling


# def whole_hist_viz(image, max_thres, min_thres, name, slice_id):
#     """
#     histogram drawing for each slice to determine the suitable threshold
#     for segmentation
#     """
#     MIN_BOUND = -1000
#     MAX_BOUND = 800
#     if max_thres < MAX_BOUND:
#         MAX_BOUND = max_thres
#     if min_thres > MIN_BOUND:
#         MIN_BOUND = min_thres
#     image[image > MAX_BOUND] = MAX_BOUND
#     image[image < MIN_BOUND] = MIN_BOUND
#     plt.hist(image.ravel(), max_thres - min_thres, [min_thres, max_thres], density=True)
#     plt.title(name + '_slice' + slice_id, fontsize='large', fontweight='bold')
#     plt.show()


def img_windowing(image, max_thres, min_thres, lung=True):
    # normalize pixels to 0 ~ 1
    if lung is True:
        min_bound = -1500.0
        max_bound = 500.0
        # if max_thres < max_bound:
        #     max_bound = max_thres
        # if min_thres > min_bound:
        #     min_bound = min_thres
        image[image > max_bound] = max_bound
        image[image < min_bound] = min_bound
        image = np.uint8((image - min_bound) / (max_bound - min_bound) * 255)
        _, seg_thres = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        min_bound = -145.0
        max_bound = 235.0
        # if max_thres < max_bound:
        #     max_bound = max_thres
        # if min_thres > min_bound:
        #     min_bound = min_thres
        image[image > max_bound] = max_bound
        image[image < min_bound] = min_bound
        image = np.uint8((image - min_bound) / (max_bound - min_bound) * 255)
        _, seg_thres = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image, seg_thres


def image_masked(img_mask, img):
    """
    :param img_mask: a binary mask to hide the background
    :param img: input image
    :return: a background-masked image
    """
    img = np.array(img)
    # print(img)
    img_mask = np.array(img_mask)
    # print(img_mask)
    return img_mask * img


def segmentation_interface(mhd_dir, out_dir, button):

    mhd_path_list = tool_packages.get_path(mhd_dir, 'mhd')
    length = len(mhd_path_list)
    count = 0
    for path in mhd_path_list:
        # a = time.time()
        img_set, _, __ = tool_packages.raw_image_reader(path)
        file_name = tool_packages.get_filename(path)
        np_array, fname, slicenum = image_segmentor(img_set, file_name, button=button)
        tool_packages.raw_image_writer(np_array, out_dir + fname + '.mhd')
        # b = time.time()
        # print('elapse: ' + str(b-a))
        # print('slices:' + str(slicenum))
        count += 1
        print('file processed: ' + str(count) + '/' + str(length))
    # print(samples)

    return None


def image_segmentor(image_array_3d, button):
    """
    Utility: image segmentation for lung CT scan,
    params:
        mhdfile_path -> the path to mhd file
    returns:
        mask array
    """
    img_set = image_array_3d.copy()
    slice_num, width, height = img_set.shape
    rt = []
    # print(slice_num)
    for i in range(slice_num):
        image = np.squeeze(img_set[i])

        # max_pixel_value = image.max()
        # min_pixel_value = image.min()

        image, segment_threshold = img_windowing(image, 0, 0, lung=button)

        # im2, contours, _ = cv2.findContours(segment_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
        #
        # cv2.imshow("img", image)
        # cv2.waitKey(0)

        if button is True:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            opening = cv2.morphologyEx(segment_threshold, cv2.MORPH_OPEN, kernel)
            # closing = cv2.morphologyEx(segment_threshold, cv2.MORPH_CLOSE, kernel)

            lca = largest_connect_area(opening)
            reversed_lca = largest_connect_area(binary_img_reverse(lca))
            # print(reversed_lca[256][256])
            if reversed_lca[256][256] == False:
                reversed_lca = binary_img_reverse(reversed_lca)
                # print(reversed_lca)
            # reversed_lca = binary_img_reverse(lca)

            result_ = image_masked(reversed_lca, image)
        else:
            result_ = image
        """
        plt visualization below
        
        """
        plt.figure()
        plt.subplot(1, 2, 1), plt.imshow(image, 'gray'), plt.title('original')
        plt.subplot(1, 2, 2), plt.imshow(result_, 'gray'), plt.title('result')
        # plt.subplot(2, 2, 3), plt.imshow(opening, 'gray'), plt.title('opening')
        # plt.subplot(2, 2, 4), plt.imshow(lca, 'gray'), plt.title('lca')
        plt.show()
        if img_set.shape[1] != 512:
            img_resize = cv2.resize(result_, (512, 512))
            rt.append(img_resize)
        else:
            rt.append(result_)
    return rt, slice_num


if __name__ == '__main__':
    samples = '../chestCT_round1/test/320831.mhd'
    img_set, _, __ = tool_packages.raw_image_reader(samples)
    file_name = tool_packages.get_filename(samples)
    result, fn, s_num = image_segmentor(img_set, file_name, True)  # true->lung. false->muscle, tissue, bone
    # plt.figure()
    # plt.subplot(2, 2, 1),
    # plt.imshow(result[10], 'gray'), plt.title('test1')
    # plt.subplot(2, 2, 2), plt.imshow(result[5], 'gray'), plt.title('test2')
    # plt.subplot(2, 2, 3), plt.imshow(result[10], 'gray'), plt.title('test3')
    # plt.subplot(2, 2, 4), plt.imshow(result[15], 'gray'), plt.title('test4')
    # plt.show()
