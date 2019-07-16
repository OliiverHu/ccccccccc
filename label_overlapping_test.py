import preprocess.tool_packages as tool_packages
import utils.bbox as bbox


def get_mhd_directly(mhd_path):
    file = open(mhd_path, 'r')
    lines = file.readlines()
    # for line in lines[6] + lines[9]:
    origin_ = lines[6].split(' = ')[1].replace('\n', "")
    spacing_ = lines[9].split(' = ')[1].replace('\n', "")
    origin_ = list(origin_.split(' '))
    spacing_ = list(spacing_.split(' '))
    for j in range(len(origin_)):
        origin_[j] = float(origin_[j])
    for j in range(len(spacing_)):
        spacing_[j] = float(spacing_[j])
    return [origin_[0], origin_[1], origin_[2], spacing_[0], spacing_[1], spacing_[2]]


annotation = tool_packages.read_csv('do_not_git/chestCT_round1_annotation.csv')
labels = tool_packages.get_label_coords(annotation, '', overlapping_test=True)
# print(labels[0])

linux_dir_path = ['E:/tianchi-chestCT/chestCT_round1/train_part1/', 'E:/tianchi-chestCT/chestCT_round1/train_part2/']
                  # '/home/huyunfei/ct_scan/ct_data/train_part3/', '/home/huyunfei/ct_scan/ct_data/train_part4/',
                  # '/home/huyunfei/ct_scan/ct_data/train_part5/']

mhd_path_list = []
for path in linux_dir_path:
    tmp = tool_packages.get_path(path, 'mhd')
    mhd_path_list += tmp

length = len(mhd_path_list)
label_db = []
item = []
for path in mhd_path_list:
    file_name = tool_packages.get_filename(path)
    origin_spacing_list = get_mhd_directly(path)
    item.append((file_name, origin_spacing_list))

dictionary = dict(item)


label_dict, name_list = tool_packages.get_label_dict(labels, mhd_path_list, translator=True)

# print(label_dict.keys())
# txt_file = open('test.txt', 'a')
max = 0.05
for name in name_list:
    boxes = label_dict[name]
    # txt_file.writelines("filename:" + name + "\n")
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            if bbox._interval_overlap([boxes[i][2], boxes[i][5]], [boxes[j][2], boxes[j][5]]) == 0:
                pass
            else:
                b1 = bbox.BoundBox(boxes[i][0], boxes[i][1], boxes[i][3], boxes[i][4])
                b2 = bbox.BoundBox(boxes[j][0], boxes[j][1], boxes[j][3], boxes[j][4])
                iou = bbox.bbox_iou(b1, b2)
                if iou > max:
                    max = iou
                # if iou != 0 and boxes[i][-1] == boxes[j][-1]:
                #     txt_file.writelines("IOU = " + str(iou) + ", " +
                #                         "box1:[" + str(boxes[i][0]) + ", " + str(boxes[i][1]) + ", " +
                #                         str(boxes[i][3]) + ", " + str(boxes[i][4]) + "] "
                #                         "box2:[" + str(boxes[j][0]) + ", " + str(boxes[j][1]) + ", " +
                #                         str(boxes[j][3]) + ", " + str(boxes[j][4]) + "]\n")

print(max)
