import numpy as np
import matplotlib.pyplot as plt


class DATA():
    def __init__(self):
        pass

    def read_data(self, file_path):
        with open(file_path, 'r') as f:
            print('read:' + file_path)
            r = f.read()
            data_list = r.split('\n')
            for k in range(len(data_list)):
                if data_list[k].find('Epoch') != -1:
                    data_out = data_list[k:]
                    return data_out
            print('error')
            return None

    def creat_dict(self):
        one_batch = {'batch_size': 10,
                     'step_in_batch': 0,
                     'yolo_layer_1_loss': 0.0,
                     'yolo_layer_2_loss': 0.0,
                     'yolo_layer_3_loss': 0.0,
                     'loss': 0.0,
                     '13': {'avg_obj': 0.0,
                            'avg_noobj': 0.0,
                            'avg_iou': 0.0,
                            'avg_cat': 0.0,
                            'recall50': 0.0,
                            'recall75': 0.0,
                            'count': 0.0,
                            'loss': {'xy': 0.0,
                                     'wh': 0.0,
                                     'conf': 0.0,
                                     'class': 0.0}},
                     '26': {'avg_obj': 0.0,
                            'avg_noobj': 0.0,
                            'avg_iou': 0.0,
                            'avg_cat': 0.0,
                            'recall50': 0.0,
                            'recall75': 0.0,
                            'count': 0.0,
                            'loss': {'xy': 0.0,
                                     'wh': 0.0,
                                     'conf': 0.0,
                                     'class': 0.0}},
                     '52': {'avg_obj': 0.0,
                            'avg_noobj': 0.0,
                            'avg_iou': 0.0,
                            'avg_cat': 0.0,
                            'recall50': 0.0,
                            'recall75': 0.0,
                            'count': 0.0,
                            'loss': {'xy': 0.0,
                                     'wh': 0.0,
                                     'conf': 0.0,
                                     'class': 0.0}}
                     }
        return one_batch

    def deal_data(self, train_data, data_in):
        layers = ['13', '26', '52']
        names = ['avg_obj', 'avg_noobj', 'avg_iou', 'avg_cat', 'recall50', 'recall75', 'count']
        loss_name = ['loss', 'yolo_layer_1_loss', 'yolo_layer_2_loss', 'yolo_layer_3_loss']
        one_batch = self.creat_dict()
        for k in range(len(data_in)):

            index = data_in[k].find('yolo_layer_1_loss')
            if index != -1:
                index = data_in[k].find('/')
                cut_data = data_in[k][:index]
                one_batch['step_in_batch'] = int(cut_data)

                index = data_in[k].find('avg_obj')
                if index != -1:
                    cut_data = data_in[k][:index]
                else:
                    cut_data = data_in[k]

                index = cut_data.find('loss')
                cut_data = cut_data[index:]
                cut_data = cut_data.split('-')

                for loss_i in range(len(cut_data)):
                    loss_data = (cut_data[loss_i].split(':'))[1]
                    try:
                        one_batch[loss_name[loss_i]] = float(loss_data)
                    except ValueError:
                        print('loss_data error!')
                        continue

                train_data.append(one_batch)
                one_batch = self.creat_dict()

            for name in names:
                for layer in layers:
                    index = data_in[k].find(name)
                    if index != -1:
                        cut_data = data_in[k][index:]
                        cut_data = self.str2float(cut_data, layer)
                        if cut_data is None or cut_data == []:
                            continue
                        one_batch[layer][name] = cut_data[0]
                        break

                    index = data_in[k].find('loss xy, wh, conf, class')
                    if index != -1:
                        cut_data = data_in[k][index:]
                        cut_data = self.str2float(cut_data, layer)
                        if cut_data is None:
                            continue
                        one_batch[layer]['loss']['xy'] = cut_data[0]
                        one_batch[layer]['loss']['wh'] = cut_data[1]
                        one_batch[layer]['loss']['conf'] = cut_data[2]
                        one_batch[layer]['loss']['class'] = cut_data[3]
                        break

        return train_data

    def str2float(self, cut_data, layer):
        output = []
        layer = '[' + layer + ']'
        index = cut_data.find(layer)
        if index == -1:
            return None
        cut_data = cut_data[index + 5:]
        cut_data = cut_data.split(']')
        for i in range(len(cut_data)):
            try:
                output.append(float(cut_data[i]))
            except ValueError:
                try:
                    output.append(float(cut_data[i][1:]))
                except ValueError:
                    continue
                continue
        return output


def draw_point(x, y, ax, dx, dy):
    ax.plot(x, y, "ro")
#   s为需要显示的字符串，xy为箭头需要指向的位置，xytext为文本的位置（个人理解是xy的相对位置）
    ax.annotate(s="({},{})".format(x, round(y, 4)), xy=(x, y), xycoords="data", xytext=(dx, dy),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))


def draw_fig(x_step, y, label, mean=0):
    fig = plt.figure(figsize=(8, 6))

    x_show = x_step.copy()
    y_show = y.copy()

    if mean > 1:
        mean_step = mean
        y_show = []
        for k in range(len(y)):
            add = []
            mean_sum = 0
            mean_num = 0
            for i in range(len(y[k])):
                mean_sum += y[k][i]
                mean_num += 1
                if i % mean_step == mean_step - 1 or i == len(y[k]):
                    add.append(mean_sum/mean_num)
                    mean_sum = 0
                    mean_num = 0
            y_show.append(add)
        add = []
        mean_sum = 0
        mean_num = 0
        for i in range(len(x_step)):
            mean_sum += x_step[i]
            mean_num += 1
            if i % mean_step == mean_step - 1 or i == len(x_step):
                add.append(mean_sum / mean_num)
                mean_sum = 0
                mean_num = 0
        x_show = add.copy()

    ax1 = fig.add_subplot(111)
    for k in range(len(y_show)):
        ax1.plot(x_show, y_show[k], label=label[k])
        # ax1.set_ylabel(label[k])

    ax1.set_xlim(0)
    ax1.set_xlabel("step")
    ax1.set_ylabel("")
    plt.legend(loc='best')

    plt.grid()  # 生成网格线
    plt.show()

    return


def draw_fig2(x_step, y1_loss, y2_precision):
    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA

    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    # par2 = host.twinx()

    # offset = 100
    # new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    # par2.axis["right"] = new_fixed_axis(loc="right",
    #                                     axes=par2,
    #                                     offset=(offset, 0))

    par1.axis["right"].toggle(all=True)
    # par2.axis["right"].toggle(all=True)

    # host.set_xlim(0, 2)
    # host.set_ylim(0, 2)

    host.set_xlabel("Step")
    host.set_ylabel("Loss")
    par1.set_ylabel("Precision")
    # par2.set_ylabel("Velocity")

    p1, = host.plot(x_step, y1_loss, label="Loss")
    p2, = par1.plot(x_step, y2_precision, label="Precision")
    # p3, = par2.plot([0, 1, 2], [50, 30, 15], label="Velocity")

    # par1.set_ylim(0, 2.5)
    # par2.set_ylim(1, 65)

    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    # par2.axis["right"].label.set_color(p3.get_color())

    plt.draw()
    plt.show()


def final_mean(y_ini):
    mean_step = 50
    y_mean = []
    for k in range(len(y_ini)):
        mean_sum = 0
        mean_num = 0
        y_cut = y_ini[k][-mean_step:]
        for i in range(len(y_cut)):
            mean_sum += y_cut[i]
            mean_num += 1
        y_mean.append(mean_sum / mean_num)

    return y_mean


def create_y(train_data, name, loss=False):
    y_out = []
    if loss:
        y_add = [train_data[i]['13']['loss'][name] for i in range(len(train_data))]
        label.append('loss_' + name + '_13')
        y_out.append(y_add)
        y_add = [train_data[i]['26']['loss'][name] for i in range(len(train_data))]
        label.append('loss_' + name + '_26')
        y_out.append(y_add)
        y_add = [train_data[i]['52']['loss'][name] for i in range(len(train_data))]
        label.append('loss_' + name + '_52')
        y_out.append(y_add)
    else:
        y_add = [train_data[i]['13'][name] for i in range(len(train_data))]
        label.append(name + '_13')
        y_out.append(y_add)
        y_add = [train_data[i]['26'][name] for i in range(len(train_data))]
        label.append(name + '_26')
        y_out.append(y_add)
        y_add = [train_data[i]['52'][name] for i in range(len(train_data))]
        label.append(name + '_52')
        y_out.append(y_add)
    return y_out


def create_y_special(train_data):
    y_out = []

    y_add = [train_data[i]['yolo_layer_1_loss'] for i in range(len(train_data))]
    label.append('yolo_layer_1_loss')
    y_out.append(y_add)
    y_add = [train_data[i]['yolo_layer_2_loss'] for i in range(len(train_data))]
    label.append('yolo_layer_2_loss')
    y_out.append(y_add)
    y_add = [train_data[i]['yolo_layer_3_loss'] for i in range(len(train_data))]
    label.append('yolo_layer_3_loss')
    y_out.append(y_add)
    y_add = [train_data[i]['loss'] for i in range(len(train_data))]
    label.append('loss')
    y_out.append(y_add)

    return y_out


if __name__ == '__main__':

    data = DATA()

    train_data = []
    file_path = '备份/20190525_06E2/printE2.txt'
    data_out = data.read_data(file_path)
    train_data = data.deal_data(train_data, data_out)

    label = []
    x_step = [i for i in range(len(train_data))]

    # y = create_y(train_data, 'class', True)
    # y = create_y(train_data, 'avg_iou', False)
    y = create_y_special(train_data)

    y_final_mean = final_mean(y)
    print(y_final_mean)

    draw_fig(x_step, y, label, mean=296)
