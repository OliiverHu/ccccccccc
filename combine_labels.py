import numpy as np
from utils.bbox import BoundBox, bbox_iou
import os
from preprocess.tool_packages import get_mhd_directly
from preprocess.coordinates_translator import pix2mm


def cal_conf(data_list):
    return np.mean(data_list)


class Labels3D():
    def __init__(self, thresh_inf_sup=0.5, thresh_iou=0.5):
        self.thresh_inf_sup = thresh_inf_sup
        self.thresh_iou = thresh_iou

    def get_data(self, filename):
        out_data = []
        with open(filename) as f:
            data = f.read()
            data = data.split('\n')
            for label in data:
                if len(label) < 10:
                    continue
                label = label.split(' ')
                if len(label) > 8:
                    label = label[:8]
                    # print('error label!>8')
                elif len(label) < 8:
                    print('error label!<8')
                    continue

                dic_label = self.to_dict(label, filename)

                out_data += [dic_label]

        return out_data

    def to_dict(self, label, filename):
        dic_label = dict()
        dic_label['class'] = int(label[0])
        dic_label['conf'] = float(label[1])
        dic_label['xmin'] = int(label[2])
        dic_label['ymin'] = int(label[3])
        dic_label['xmax'] = int(label[4])
        dic_label['ymax'] = int(label[5])
        dic_label['inf'] = float(label[6])
        dic_label['sup'] = float(label[7])

        filename = filename.split('/')[-1]
        filename = filename.split('.')[0]
        filename = filename.split('_')[-1]
        dic_label['z'] = int(filename)

        dic_label['finish_inf'] = False
        dic_label['finish_sup'] = False

        return dic_label

    def get_all_data(self, file_list):
        labels = []
        labels_3D = []
        for one_file in file_list:
            labels += self.get_data(one_file)

        for label in labels:
            label_3D = []
            next_label = label
            while next_label is not None:
                label_3D += [next_label]
                next_label = self.find_next('inf', next_label, labels)
            while next_label is not None:
                label_3D += [next_label]
                next_label = self.find_next('sup', next_label, labels)
            labels_3D += [label_3D]

        return labels_3D

    def find_next(self, inf_or_sup, present, labels):
        if inf_or_sup == 'inf':
            if present['finish_inf']:
                return None
            for label in labels:
                if label['finish_sup']:
                    continue
                if label['class'] != present['class']:
                    continue
                if label['z'] != present['z'] - 1:
                    continue
                if not self.compare(label, present):
                    continue
                if label['sup'] < self.thresh_inf_sup and present['inf'] < self.thresh_inf_sup:
                    continue
                label['finish_sup'] = True
                present['finish_inf'] = True
                return label

            if present['inf'] > self.thresh_inf_sup:
                label = present.copy()
                label['z'] = present['z'] - 1
                label['finish_sup'] = True
                label['finish_inf'] = True
                present['finish_inf'] = True
                return label

            present['finish_inf'] = True
            return None

        else:
            if present['finish_sup']:
                return None
            for label in labels:
                if label['finish_inf']:
                    continue
                if label['class'] != present['class']:
                    continue
                if label['z'] != present['z'] + 1:
                    continue
                if not self.compare(label, present):
                    continue
                if label['inf'] < self.thresh_inf_sup and present['sup'] < self.thresh_inf_sup:
                    continue
                label['finish_inf'] = True
                present['finish_sup'] = True
                return label

            if present['sup'] > self.thresh_inf_sup:
                label = present.copy()
                label['z'] = present['z'] + 1
                label['finish_sup'] = True
                label['finish_inf'] = True
                present['finish_sup'] = True
                return label

            present['finish_sup'] = True
            return None

    def compare(self, data1, data2):
        if data2['xmin'] <= data1['xmin'] <= data1['xmax'] <= data2['xmax'] \
           and data2['ymin'] <= data1['ymin'] <= data1['ymax'] <= data2['ymax']:
            return True
        if data1['xmin'] <= data2['xmin'] <= data2['xmax'] <= data1['xmax'] \
           and data1['ymin'] <= data2['ymin'] <= data2['ymax'] <= data1['ymax']:
            return True
        box1 = BoundBox(data1['xmin'], data1['ymin'], data1['xmax'], data1['ymax'])
        box2 = BoundBox(data2['xmin'], data2['ymin'], data2['xmax'], data2['ymax'])

        iou = bbox_iou(box1, box2)

        if iou > self.thresh_iou:
            return True
        else:
            return False

    def build_3D(self, label_3D):
        xmax = max([label['xmax'] for label in label_3D])
        xmin = min([label['xmin'] for label in label_3D])
        ymax = max([label['ymax'] for label in label_3D])
        ymin = min([label['ymin'] for label in label_3D])
        zmax = max([label['z'] for label in label_3D])
        zmin = min([label['z'] for label in label_3D])
        conf = cal_conf([label['conf'] for label in label_3D])
        label_class = label_3D[0]['class']

        return xmax, xmin, ymax, ymin, zmax, zmin, label_class, conf

    def tran_data(self, data, mhd_name):
        position = get_mhd_directly(mhd_name)
        origin = position[:3]
        spacing = position[3:]
        point0 = [(data[0]+data[1])/2, (data[2]+data[3])/2, (data[4]+data[5])/2]
        whl0 = [data[0]-data[1], data[2]-data[3], data[4]-data[5]]
        point0, whl0 = pix2mm(point0, whl0, origin, spacing)
        return point0[0], point0[1], point0[2], data[6], data[7]

    def out_put(self, labels_3D, txt_name, mhd_name):
        newline = ''
        for label_3D in labels_3D:
            data = self.build_3D(label_3D)
            data = self.tran_data(data, mhd_name)
            newline += ' %.6f %.6f %.6f %d %.6f \n' % data
        with open(txt_name, 'w') as f:
            f.write(newline)


if __name__ == '__main__':
    txt_dir = 'E:/Training/chestCT/test_output/'
    out_dir = 'E:/Training/chestCT/test_output_3D/'
    mhd_dir = 'E:/Training/chestCT/test_input/'
    txt_paths = []
    for inp_file in os.listdir(txt_dir):
        txt_paths += [txt_dir + inp_file]

    label_paths = [inp_file for inp_file in txt_paths if (inp_file[-4:] in ['.txt'])]

    seen_raw = dict()
    for label_path in label_paths:
        filename = label_path.split('/')[-1]
        filename = filename.split('_')[0]
        if filename in seen_raw:
            seen_raw[filename] += [label_path]
        else:
            seen_raw[filename] = [label_path]

    for raw in seen_raw:
        Func = Labels3D(0.5, 0.5)
        Func.out_put(Func.get_all_data(seen_raw[raw]), out_dir + raw + '.txt', mhd_dir + raw + '.mhd')