from preprocess.label_parser import file_parser_interface
from preprocess.tool_packages import get_path, read_csv


def label_generation():
    linux_dir_path = ['/home/huyunfei/ct_scan/ct_data/train_part1/', '/home/huyunfei/ct_scan/ct_data/train_part2/',
                      '/home/huyunfei/ct_scan/ct_data/train_part3/', '/home/huyunfei/ct_scan/ct_data/train_part4/',
                      '/home/huyunfei/ct_scan/ct_data/train_part5/']

    win_dir_path = ['chestCT_round1/test/']

    annotation_path = 'do_not_git/chestCT_round1_annotation.csv'
    csv_file_handle = read_csv(annotation_path)
    # out_dir = '/home/huyunfei/ct_scan/processed_data/'
    out_dir = ''

    mhd_path_list = []
    for path in win_dir_path:
        tmp = get_path(path, 'mhd')
        mhd_path_list += tmp

    file_parser_interface(mhd_path_list, csv_file_handle, out_dir)
