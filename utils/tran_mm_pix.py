import numpy as np


def pix2mm(point, whl, origin, spacing):
    point = np.array(point, np.float)
    whl = np.array(whl, np.float)
    origin = np.array(origin, np.float)
    spacing = np.array(spacing, np.float)
    out_whl = whl*spacing
    out_point = point*spacing + origin
    return list(out_point), list(out_whl)


def mm2pix(point, whl, origin, spacing):
    point = np.array(point, np.float)
    whl = np.array(whl, np.float)
    origin = np.array(origin, np.float)
    spacing = np.array(spacing, np.float)
    out_whl = whl/spacing
    out_point = (point-origin)/spacing
    return list(out_point), list(out_whl)


if __name__ == '__main__':
    point1 = [1, 2, 3]
    point1 = np.array(point1)
    whl1 = [1, 2, 3]

    origin1 = [1, 2, 3]
    spacing1 = [1, 2, 3]
    point2, whl2 = pix2mm(point1, whl1, origin1, spacing1)
    print(point1)
    print(point2)
    print(whl2)