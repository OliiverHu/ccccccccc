import preprocess.tool_packages as tool_packages
import preprocess.img_seg as img_seg
import numpy as np


def raw_reader(path, i, lung=True):
    """
    :param path: path to mhd file
            i: the id of slice in raw file
            lung=True: lung windowing
            lung=False: bone,tissue,muscle windowing
    :return: a 3 channel numpy array
    """
    # file_name = tool_packages.get_filename(path)
    img_set, origin, spacing = tool_packages.raw_image_reader(path)
    masked_img = []
    if i == 0:
        masked_img.append(img_set[0])
        masked_img.append(img_set[0])
        masked_img.append(img_set[1])
    elif i == len(img_set)-1:
        masked_img.append(img_set[-2])
        masked_img.append(img_set[-1])
        masked_img.append(img_set[-1])
    else:
        for j in range(i - 1, i + 2, 1):
            image = np.squeeze(img_set[j])
            masked_img.append(image)

    masked_img, _slice_num = img_seg.image_segmentor(np.array(masked_img), lung)

    three_channels = np.array(masked_img).transpose((1, 2, 0))

    return three_channels
