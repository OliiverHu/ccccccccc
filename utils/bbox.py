import numpy as np
import matplotlib.pyplot as plt


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None, inf = 1, sup = 1):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.inf = inf
        self.sup = sup

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score      


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3    


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    try:
        return float(intersect) / union
    except ZeroDivisionError:
        return 0


def draw_boxes(image, boxes, labels, obj_thresh, path, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1
        plt.imshow(image, cmap='gray')
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '':
                    label_str += ', '
                label_str += (labels[i])# + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet:
                print(label_str)
                
        if label >= 0:
            plt.gca().add_patch(plt.Rectangle(xy=(box.xmin, box.ymin), width=int(box.xmax - box.xmin),
                                              height=int(box.ymax - box.ymin), edgecolor='#FF0000',
                                              fill=False, linewidth=0.5))

            plt.text(int(box.xmin), int(box.ymin) - 10, label_str, size=10, family="fantasy", color="r",
                     style="italic", weight="light")

    plt.savefig(path)
    [p.remove() for p in reversed(plt.gca().patches)]
    [p.remove() for p in reversed(plt.gca().texts)]



            # text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 3)
            # width, height = text_size[0][0], text_size[0][1]
            # region = np.array([[box.xmin-3,        box.ymin],
            #                    [box.xmin-3,        box.ymin-height-26],
            #                    [box.xmin+width+13, box.ymin-height-26],
            #                    [box.xmin+width+13, box.ymin]], dtype='int32')
            #
            # cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=2)
            # cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            # cv2.putText(img=image,
            #             text=label_str,
            #             org=(box.xmin+13, box.ymin - 13),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=3e-4 * image.shape[0],
            #             color=(0,0,0),
            #             thickness=1)
        
    return image


def get_box_info(line, boxes, labels, obj_thresh):
    for box in boxes:
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                # if label_str != '': label_str += ', '
                # label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                line += labels[i]
                label = i
        if box.xmin < 0:
            box.xmin = 0
        if box.ymin < 0:
            box.ymin = 0

        if label >= 0:
            line += ' %.6f %d %d %d %d %.6f %.6f \n'% \
                    (box.get_score(), box.xmin, box.ymin, box.xmax, box.ymax, box.inf, box.sup)

    newline = line[:-1]
    return newline


