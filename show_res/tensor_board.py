from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

#加载日志数据
# ea = event_accumulator.EventAccumulator(r'E:/laboratory/project/YOLOv3/log_voc/events.out.tfevents.1555646582'
#                                         r'.USER-20140825OW')
# ea = event_accumulator.EventAccumulator(r'E:/laboratory/project/YOLOv3/log_voc/events.out.tfevents.1558485415'
#                                         r'.USER-20140825OW')


def get_data(name, filename):
    # ea = event_accumulator.EventAccumulator('E:/Project/ChestCT/log_voc/events.out.tfevents.'
    #                                         + filename + '.user-System-Product-Name')
    ea = event_accumulator.EventAccumulator('E:/Project/ChestCT/log_voc/events.out.tfevents.'
                                            + filename + '.DESKTOP-50KJA8U')
    ea.Reload()
    print(ea.scalars.Keys())

    data = ea.scalars.Items(name)
    print(len(data))
    # print([(i.step, i.value) for i in data])
    # loss = [(i.step, i.value) for i in loss]
    for k in range(1, len(data)):
        try:
            if data[k].step <= data[k-1].step:
                data.pop(k)
                # print(k)
        except IndexError:
            break
    # print([(i.step, i.value) for i in loss])
    # print(len(loss))
    return data


def do_mean(data, mean=0):
    mean_step = mean
    y_show = []
    y = data.copy()


    add = []
    mean_sum = 0
    mean_num = 0
    for i in range(len(y)):
        mean_sum += y[i].value
        mean_num += 1
        if i % mean_step == mean_step - 1 or i == len(y):
            add.append(mean_sum / mean_num)
            mean_sum = 0
            mean_num = 0
    y_show = add
    return y_show


def get_data_all(name):
    data = []

    data += get_data(name, '1563002218')
    data += get_data(name, '1563024998')
    # data += get_data(name, '1560907058')
    # data += get_data(name, '1560915190')
    # data += get_data(name, '1562916351')
    # data += get_data(name, '1562917323')
    # data += get_data(name, '1562917787')
    # data += get_data(name, '1562921929')

    return data


def plot_line(name, axi):
    data = get_data_all(name)
    if name == 'lr':
        y = do_mean(data, 1)
        y = np.log10(y)
        print(y)
    else:
        y = do_mean(data, 100)
    # axi.plot([i.step for i in data], [i.value for i in data], label=name)
    axi.plot([i*100 for i in range(len(y))], y, label=name)


fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)

plot_line('loss', ax1)
plot_line('yolo_layer_1_loss', ax1)
plot_line('yolo_layer_2_loss', ax1)
plot_line('yolo_layer_3_loss', ax1)
# plot_line('lr', ax1)

ax1.set_xlim(0)
ax1.set_xlabel("step")
ax1.set_ylabel("")

plt.legend(loc='upper right')
plt.show()