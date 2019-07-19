import preprocess.tool_packages as tool_packages
import preprocess.coordinates_translator as translator
import numpy as np
import os


def file_parser_interface(mhdfile_path_list, out_path, csv_handle):
    """
    :Function a raw format file parser, to generate npy files and corresponding labels for dl training
    :param mhdfile_path_list: a list of mhd file path
            anno_path: path to the annotation file
    :return: None
    """
    length = len(mhdfile_path_list)
    count = 0
    for path in mhdfile_path_list:
        file_name = tool_packages.get_filename(path)
        img_set, origin, spacing = tool_packages.raw_image_reader(path)
        slice_num, width, height = img_set.shape  # can be optimized HERE
        # param1 = 1
        # if width != 512:
        #     param1 = label_normalization_parameter(width)
        for i in range(slice_num):
            label_parser(out_path + file_name, i, path, slice_num-1, csv_handle)

        count += 1
        print('file processed: ' + str(count) + '/' + str(length))


def label_parser(file_name, slice_num, def_path, slice_max, csv_handle):
    """
    :param file_name: ouput file path
    :param
    :param
    :param slice_num: which slice of raw file
    :param def_path: mhd file path
    :param
    :param
    :return: None
    """
    # csv_file = tool_packages.read_csv(annotation_path)
    ab_name = tool_packages.get_filename(file_name)
    labels = tool_packages.get_label_pixel_coords(ab_name, csv_handle)
    # labels = tool_packages.get_label_world_coords(csv_file_handle, ab_name)
    if len(labels) == 0:
        return None

    label_db = []
    for label in labels:
        if label[6] == label[3]:  # if this label only has one layer
            # print(min_coord[2])
            if int(label[3]) == slice_num:
                if slice_num == 0:
                    label_db.append([label[1], label[2], label[4], label[5], label[7], 0, 1])
                    continue
                elif slice_num == slice_max:
                    label_db.append([label[1], label[2], label[4], label[5], label[7], 1, 0])
                else:
                    label_db.append([label[1], label[2], label[4], label[5], label[7], 0, 0])
                # print('received')
            else:
                pass
        else:  # if this label has multiple layers
            for i in range(int(label[3]), int(label[6]) + 1, 1):
                if i == slice_num == 0:
                    label_db.append([label[1], label[2], label[4], label[5], label[7], 1, 1])
                    continue
                elif i == slice_num == slice_max:
                    label_db.append([label[1], label[2], label[4], label[5], label[7], 1, 1])
                else:
                    if i == slice_num:
                        # print('received')
                        if i == int(label[3]):
                            label_db.append([label[1], label[2], label[4], label[5], label[7], 1, 0])
                        elif i == int(label[6]):
                            label_db.append([label[1], label[2], label[4], label[5], label[7], 0, 1])
                        else:
                            label_db.append([label[1], label[2], label[4], label[5], label[7], 1, 1])
                    else:
                        pass

    header = 'obj ' + 'xmin ymin xmax ymax ' + 'inferior superior ' + 'class '
    n = file_name + '_slice' + str(slice_num+1)
    file = open(n + '.txt', 'a')
    file.write(str(slice_num))
    file.write('\n')
    def_path = def_path.replace('.mhd', '.raw')
    file.write(def_path)
    file.write('\n')
    file.write(header)
    file.write('\n')
    if len(label_db) == 0:
        file.write('0')
    else:
        flag = 1
        for label in label_db:
            file.write(str(flag) + ' ')
            file.write(str(label[0]) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + ' ' +
                       str(label[5]) + ' ' + str(label[6]) + ' ' + str(label[4]))
            file.write('\n')
            flag += 1

    file.close()
    return None


def annotation_csv_translator(csv_file_path, out_path):
    lines = tool_packages.read_csv(csv_file_path)
    tool_packages.write_csv(out_path, ['seriesuid', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'label'])
    linux_dir_path = ['/home/huyunfei/ct_scan/ct_data/train_part1/', '/home/huyunfei/ct_scan/ct_data/train_part2/',
                      '/home/huyunfei/ct_scan/ct_data/train_part3/', '/home/huyunfei/ct_scan/ct_data/train_part4/',
                      '/home/huyunfei/ct_scan/ct_data/train_part5/']
    win_dir_path = ['E:/tianchi-chestCT/chestCT_round1/train_part1/']
    mhd_path_list = []
    mhd_name_item = []
    mhd_name_list = []
    for path in win_dir_path:
        tmp = tool_packages.get_path(path, 'mhd')
        mhd_path_list += tmp
    for path in mhd_path_list:
        mhd_name_item.append((tool_packages.get_filename(path), path))
        mhd_name_list.append(tool_packages.get_filename(path))
    mhd_name_dict = dict(mhd_name_item)

    for line in lines:
        if line[0] not in mhd_name_list:
            pass
        else:
            world_coord = np.asarray([float(line[1]), float(line[2]), float(line[3])])
            diameter = np.asarray([float(line[4]), float(line[5]), float(line[6])])
            origin_spacing, dims = tool_packages.get_mhd_directly(mhd_name_dict[line[0]])
            # dim3 = dims[2].replace('\n', '')
            resize_coefficient = 1
            if dims[0] != 512:
                resize_coefficient = label_normalization_parameter(dims[0])
            max_coord, _, min_coord = translator.get_8_point(world_coord, diameter, origin_spacing[:3],
                                                             origin_spacing[3:], resize_coefficient)
            tool_packages.write_csv(out_path, [line[0], min_coord[0], min_coord[1], min_coord[2],
                                               max_coord[0], max_coord[1], max_coord[2], line[7]])
    return None


def label_normalization_parameter(w):
    coefficient = 512 / int(w)

    return coefficient


if __name__ == '__main__':
    pass
    img_set, origin, spacing = tool_packages.raw_image_reader('../chestCT_round1/test/318818.mhd')
    slices, width, height = img_set.shape
    name = tool_packages.get_filename('chestCT_round1/test/318818.mhd')
    # csv_file = tool_packages.read_csv('../do_not_git/chestCT_round1_annotation.csv')
    annotation_path_world = '../do_not_git/chestCT_round1_annotation.csv'

    annotation_path_pixel = '../chestCT_round1/test.csv'
    if os.path.exists(annotation_path_pixel):
        print('the  pixel coordinates file exists')
        csv_file_handle = tool_packages.read_csv(annotation_path_pixel)
    else:
        print('generating pixel coordinates file')
        annotation_csv_translator(annotation_path_world, annotation_path_pixel)
        print('done')
        csv_file_handle = tool_packages.read_csv(annotation_path_pixel)
    file_parser_interface(['../chestCT_round1/test/318818.mhd'], '', csv_file_handle)
