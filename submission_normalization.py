from preprocess.tool_packages import write_csv, get_path, get_filename
import csv


def txt_reader(txt_file_path):
    list = []
    handler = open(txt_file_path, 'a')
    lines = handler.readlines()
    uid = get_filename(txt_file_path)
    for line in lines:
        line = line.split(' ')
        coordx = line[0]
        coordy = line[1]
        coordz = line[2]
        class_ = line[3]
        prob = line[4]
        list.append([uid, coordx, coordy, coordz, class_, prob])
    return list


txt_dir = ''
out_dir = ''
txt_paths = get_path(txt_dir, 'txt')
csv_ = open(out_dir + 'result.csv', "a+", newline='')
writer = csv.writer(csv_, quotechar=',', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability'])
for path in txt_paths:
    input_list = txt_reader(path)
    write_csv(out_dir + 'result.csv', input_list)

