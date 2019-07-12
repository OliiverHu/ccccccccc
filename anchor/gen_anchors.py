import random
import argparse
import numpy as np
import tensorflow as tf

from voc import parse_voc_annotation
import json

def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids):
    out_string = ''

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices:
        out_string += str(int(anchors[i,0]*416)) + ',' + str(int(anchors[i,1]*416)) + ', '
            
    print(out_string[:-2])

def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def _main_(argv):
    num_anchors = args.anchors
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(['../YOLOv3/NEU-DET/TFRecords/train.tfrecords'])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'image_name': tf.FixedLenFeature([], tf.string),
            'image_width': tf.FixedLenFeature([], tf.int64),
            'image_height': tf.FixedLenFeature([], tf.int64),
            'image_depth': tf.FixedLenFeature([], tf.int64),
            'image/object_number': tf.FixedLenFeature([], tf.int64),
            'image/object/names': tf.VarLenFeature(tf.string),
            'image/object/id': tf.VarLenFeature(tf.int64),
            'image/object/xmin': tf.VarLenFeature(tf.float32),
            'image/object/xmax': tf.VarLenFeature(tf.float32),
            'image/object/ymin': tf.VarLenFeature(tf.float32),
            'image/object/ymax': tf.VarLenFeature(tf.float32),
        }
    )
    images = tf.decode_raw(features['image'], tf.uint8)
    imagenames = tf.cast(features['image_name'], tf.string)
    widths = tf.cast(features['image_width'], tf.int32)
    heights = tf.cast(features['image_height'], tf.int32)
    depths = tf.cast(features['image_depth'], tf.int32)
    objects = tf.cast(features['image/object_number'], tf.int32)
    names = tf.cast(features['image/object/names'], tf.string)
    ids = tf.cast(features['image/object/id'], tf.int64)
    xmins = tf.cast(features['image/object/xmin'], tf.float32)
    xmaxs = tf.cast(features['image/object/xmax'], tf.float32)
    ymins = tf.cast(features['image/object/ymin'], tf.float32)
    ymaxs = tf.cast(features['image/object/ymax'], tf.float32)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # run k_mean to find the anchors
    annotation_dims = []
    for i in range(900):
        image, imagename, width, height, depth, objectsnumber, name, id, xmin, xmax, ymin, ymax = \
            sess.run([images, imagenames, widths, heights, depths, objects, names, ids, xmins, xmaxs, ymins, ymaxs])
        print(imagename)
        for k in range(objectsnumber):
            relative_w = (float(xmax.values[k]) - float(xmin.values[k]))/width
            relatice_h = (float(ymax.values[k]) - float(ymin.values[k]))/height
            annotation_dims.append(tuple(map(float, (relative_w,relatice_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-a',
        '--anchors',
        default=9,
        help='number of anchors to use')

    args = argparser.parse_args()
    _main_(args)
