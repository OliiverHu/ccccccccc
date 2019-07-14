import numpy as np
"""
convert world coordinate to real coordinate
# input: point0     (list or np.array format)
#        whl0       (list or np.array format)
#        origin     (list or np.array format)
#        spacing    
# output:points[0]  (xyzmax)
#        points     (points)
#        points[7]  (xyzmin)
"""


def get_8_point(point0, whl0, origin, spacing, resize_coefficient):
    point0, whl0 = mm2pix(point0, whl0, origin, spacing)
    points = tran_data(point0, whl0, resize_coefficient)

    return points[0], points, points[7]


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


def tran_data(point0, whl0, resize_coefficient, tran_type=1):
    case = [[ 1, 1, 1],
            [ 1, 1,-1],
            [ 1,-1, 1],
            [ 1,-1,-1],
            [-1, 1, 1],
            [-1, 1,-1],
            [-1,-1, 1],
            [-1,-1,-1]]
    if tran_type:
        points = []
        for k in range(8):
            point = [0, 0, 0]
            for i in range(2):
                point[i] = point0[i] + case[k][i] * (whl0[i]-1)/2
                point[i] *= resize_coefficient
                if point[i] - int(point[i]) < 1E-3 or int(point[i]) - point[i] + 1 < 1E-3:
                    if point[i] < -0.5:
                        print('wrong data!(find <0)')
                    point[i] = int(point[i] + 0.1)
                else:
                    print('wrong data!(find .5)')
                    point[i] = int(point[i] + 0.5)
            i = 2
            point[i] = point0[i] + case[k][i] * (whl0[i] - 1) / 2
            if point[i] - int(point[i]) < 1E-3 or int(point[i]) - point[i] + 1 < 1E-3:
                if point[i] < -0.5:
                    print('wrong data!(find <0)')
                point[i] = int(point[i] + 0.1)
            else:
                print('wrong data!(find .5)')
                point[i] = int(point[i] + 0.5)
            points.append(point)
        return points
    else:
        pass


def voxel_diameter(dmt, spacing):
    return dmt / spacing

