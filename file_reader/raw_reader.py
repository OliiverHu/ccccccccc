import preprocess.tool_packages as tool_packages
import numpy as np


def raw_reader(path, i):
    """
    :param path: path to mhd file
            i: the id of slice in raw file
    :return: a 5 channel numpy array
    """
    # file_name = tool_packages.get_filename(path)
    img_set, origin, spacing = tool_packages.raw_image_reader(path)
    masked_img = []
    if i-2 < 0 or i+3 >img_set.shape[0]:
        return None
    for j in range(i - 2, i + 3, 1):
        image = np.squeeze(img_set[j, ...])
        masked_img.append(image)

    five_channels = np.array(masked_img).transpose((1, 2, 0))

    return five_channels

