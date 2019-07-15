from preprocess.tool_packages import write_csv, get_path, get_filename
import csv


def txt_reader(txt_file_path):
    list = []
    handler = open(txt_file_path, 'r')
    uid = get_filename(txt_file_path)
    for line in handler.readlines():
        line = line.split(' ')
        coordx = line[1]
        coordy = line[2]
        coordz = line[3]
        class_ = line[4]
        prob = line[5]
        list.append([uid, coordx, coordy, coordz, class_, prob])
    return list


txt_dir = 'E:/Training/chestCT/test_output_3D/'
out_dir = ''
txt_paths = get_path(txt_dir, 'txt')
csv_ = open(out_dir + 'result.csv', "a+", newline='')
writer = csv.writer(csv_, quotechar=',', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability'])
csv_.close()
for path in txt_paths:
    input_list = txt_reader(path)
    for list_ in input_list:
        write_csv(out_dir + 'result.csv', list_)

