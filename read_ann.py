import pickle
import os


def parse_yolo_annotation(ann_dir, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}

        for ann in sorted(os.listdir(ann_dir)):
            if ann[-4:] != '.txt':
                continue
            img = {'object': []}
            try:
                img_num, img_name, label_data = get_label_from_txt(ann_dir + ann)

                img['filename'] = img_name
                img['img_num'] = int(img_num)
                # if not os.path.exists(img['filename']):
                #     continue

                for label in label_data:
                    obj = {}
                    obj['name'] = label[7]

                    if obj['name'] in seen_labels:
                        seen_labels[obj['name']] += 1
                    else:
                        seen_labels[obj['name']] = 1

                    if len(labels) > 0 and obj['name'] not in labels:
                        pass
                    else:
                        img['object'] += [obj]

                    xmin, ymin, xmax, ymax = int(label[1]), int(label[2]), int(label[3]), int(label[4])
                    inf, sup = int(label[5]), int(label[6])
                    point0, whl0 = tran_data([xmin, ymin], [xmax, ymax], 0)
                    obj['point0'], obj['whl'] = point0, whl0
                    obj['inf'], obj['sup'] = inf, sup
                    obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'] = xmin, ymin, xmax, ymax
                    # points = tran_data(obj['point0'], obj['whl'])
                    # print(points)

            except Exception as e:
                print(e)
                print('Ignore this bad annotation: ' + ann_dir + ann)

            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels


def tran_data(data0, data1, tran_type=1):
    case = [[ 1, 1, 1],
            [ 1, 1,-1],
            [ 1,-1, 1],
            [ 1,-1,-1],
            [-1, 1, 1],
            [-1, 1,-1],
            [-1,-1, 1],
            [-1,-1,-1]]
    if tran_type:
        point0 = data0
        whl0 = data1
        points = []
        for k in range(8):
            point = [0, 0, 0]
            for i in range(3):
                point[i] = point0[i] + case[k][i] * (whl0[i]-1)/2
                if point[i] - int(point[i]) < 1E-3 or int(point[i]) - point[i] + 1 < 1E-3:
                    if point[i] < -0.5:
                        print('<0')
                    point[i] = int(point[i] + 0.1)
                else:
                    print(point0)
                    print(whl0)
                    print(point[i])
                    print('------')
            points.append(point)
        return points
    else:
        xmin, ymin = data0
        xmax, ymax = data1
        point0 = [(xmin + xmax)/2, (ymin + ymax)/2]
        whl0 = [xmax - xmin, ymax - ymin]
        return point0, whl0


def get_label_from_txt(filename):
    out_data = []
    with open(filename) as f:
        data = f.read()
        data = data.split('\n')
        img_num = data[0]
        img_name = data[1]
        data = data[3:]
        for label in data:
            if len(label) < 10:
                continue
            label = label.split(' ')
            if len(label) > 8:
                label = label[:-1]
                print('error label!>8')
            elif len(label) < 8:
                print('error label!<8')
                continue
            out_data.append(label)
    return img_num, img_name, out_data
        # print(out_data)


if __name__ == '__main__':
    xann_dir = './data/'
    ximg_dir = './data/'
    cache    = './data/cache.pkl'
    xall_insts, xseen_labels = parse_yolo_annotation(xann_dir, ximg_dir, cache, ['1', '5', '31', '32'])
    print(xseen_labels)
    print(xall_insts)


