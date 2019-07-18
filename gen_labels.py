from preprocess.label_parser import file_parser_interface
from preprocess.tool_packages import get_path, read_csv
import os

if __name__ == '__main__':
    linux_dir_path = ['/home/huyunfei/ct_scan/ct_data/train_part1/', '/home/huyunfei/ct_scan/ct_data/train_part2/',
                      '/home/huyunfei/ct_scan/ct_data/train_part3/', '/home/huyunfei/ct_scan/ct_data/train_part4/',
                      '/home/huyunfei/ct_scan/ct_data/train_part5/']

    win_dir_path = ['E:/tianchi-chestCT/chestCT_round1/train_part1/']

    annotation_path_world = 'do_not_git/chestCT_round1_annotation.csv'

    # TODO
    annotation_path_pixel = ''
    csv_file_handle = read_csv(annotation_path_world)
    # out_dir = '/home/huyunfei/ct_scan/processed_data/'
    out_dir = 'chestCT_round1/'

    mhd_path_list = []
    for path in win_dir_path:
        tmp = get_path(path, 'mhd')
        mhd_path_list += tmp

    # TODO
    # if os.path.exists()
    file_parser_interface(mhd_path_list, csv_file_handle, out_dir)
