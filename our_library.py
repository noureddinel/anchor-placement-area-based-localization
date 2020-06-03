# -*- coding: utf-8 -*-
from shapely.geometry import Point, MultiPolygon, Polygon
from numpy import *
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import shapely.speedups
from shapely.wkt import loads, dumps
from scipy.stats import entropy
from decimal import *
from tqdm import tqdm
shapely.speedups.enable()

PRECISION = 5
MAX_X = 192
MAX_Y = 192
NB_POINT_PER_SIDE = 3 # In: 3,5,9,17,33,97
TICS = MAX_X // (NB_POINT_PER_SIDE - 1)
NB_ANCHORS = 3
getcontext().prec = PRECISION


def minMax(a, b):
    if a > b:
        res = a - (2 * (a - b))
        return res
    else:
        if a == b:
            return None
        else:
            res = a + (2 * (b - a))
            return res


def drawNetwork(anchors, residence_area_l=[None], max_x_=MAX_X, max_y_=MAX_Y, nb_anchors_=NB_ANCHORS, tics_=TICS,
                algo_="zzz", mode_="file"):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    e = plt.Rectangle((0, 0), max_x_, max_y_, facecolor='w', edgecolor='black', linewidth=2.0)
    ax.add_patch(e)
    for area in residence_area_l:
        if area is None:
            continue
        if area.geom_type == "MultiPolygon":
            m_color = np.random.rand(3, )
            for poly in area:
                try:
                    ring_patch = PolygonPatch(poly, fc=m_color, alpha=0.2, zorder=2)
                    ax.add_patch(ring_patch)
                except:
                    print("Exception Drawing:1")
        if area.geom_type == "GeometryCollection":
            m_color = np.random.rand(3, )
            for poly in area:
                try:
                    ring_patch = PolygonPatch(poly, fc=m_color, alpha=0.2, zorder=2)
                    ax.add_patch(ring_patch)
                except:
                    print("Exception Drawing:2")
        else:
            if area != None:
                try:
                    ring_patch = PolygonPatch(area, fc=np.random.rand(3, ), alpha=0.2, zorder=2)
                    ax.add_patch(ring_patch)
                except:
                    print("Exception Drawing:3")
    for a in anchors:
        n = Point(a[0], a[1])
        plt.plot(n.x, n.y, '^', color='b')
    ax.grid(which='both')
    if mode_ == "file":
        plt.savefig(
            './IMG/' + str(algo_) + str(nb_anchors_) + '_anchor_' + str(max_x_) + 'x' + str(max_y_) + '_' + str(tics_),
            bbox_inches='tight')
    else:
        plt.show()
    return ax


def getAllSubRegions(anchors_, max_x_=MAX_X, max_y_=MAX_Y):
    area_list = []
    anchors_ = sorted(anchors_)
    bound_box = Polygon([(0, 0), (max_x_, 0), (max_x_, max_y_), (0, max_y_)])
    rest_of_area = bound_box
    rest_of_areaa = None

    for a1, a2 in list(combinations(anchors_, 2)):
        a1 = Point(a1[0], a1[1])
        a2 = Point(a2[0], a2[1])
        hsl_cr_list = get2anchorsHslRegions(a1, a2, max_x_, max_y_)
        for x in hsl_cr_list:
            xx = fix_shape(x)
            if xx is not None and not xx.is_empty:
                area_list.append(xx)
                try:
                    rest_of_area = rest_of_area.difference(xx)
                except:
                    pass
                rest_of_areaa = fix_shape(rest_of_area)
                rest_of_area = rest_of_areaa

    if rest_of_areaa is not None and not rest_of_areaa.is_empty:
        area_list.append(rest_of_areaa)
    else:
        area_list.append(rest_of_area)
    return area_list


def get2anchorsHslRegions(a0, a1, max_x_=MAX_X, max_y_=MAX_Y):
    bound_box = Polygon([(0, 0), (max_x_, 0), (max_x_, max_y_), (0, max_y_)])
    c1 = a0.buffer(a0.distance(a1))
    c2 = a1.buffer(a0.distance(a1))
    v = c1.intersection(c2)
    r = (c1.boundary).intersection(c2.boundary)
    m_res = []
    if not r.is_empty:
        X_first_point = minMax(r[0].x, a1.x)
        Y_first_point = minMax(r[0].y, a1.y)
        first_point = Point(X_first_point, Y_first_point)
        X_second_point = minMax(r[1].x, a1.x)
        Y_second_point = minMax(r[1].y, a1.y)
        second_point = Point(X_second_point, Y_second_point)
        polygon1 = Polygon((list(list(r)[0].coords)[0], list(list(r)[1].coords)[0], list(first_point.coords)[0],
                            list(second_point.coords)[0]))
        X_first_point = minMax(r[0].x, a0.x)
        Y_first_point = minMax(r[0].y, a0.y)
        first_point = Point(X_first_point, Y_first_point)
        X_second_point = minMax(r[1].x, a0.x)
        Y_second_point = minMax(r[1].y, a0.y)
        second_point = Point(X_second_point, Y_second_point)
        polygon2 = Polygon((list(list(r)[0].coords)[0], list(list(r)[1].coords)[0], list(first_point.coords)[0],
                            list(second_point.coords)[0]))
    try:
        hsl1 = v.intersection(polygon1.convex_hull)
        hsl1 = hsl1.intersection(bound_box)
        m_res.append(hsl1)
    except:
        print("there is a problem get2anchorsHslRegions hsl1")
    try:
        hsl2 = v.intersection(polygon2.convex_hull)
        hsl2 = hsl2.intersection(bound_box)
        m_res.append(hsl2)
    except:
        print("there is a problem get2anchorsHslRegions hsl2")
    try:
        cr1 = c1.difference(c2)
        cr1 = cr1.intersection(bound_box)
        m_res.append(cr1)
    except:
        print("there is a problem get2anchorsHslRegions cr1")
    try:
        cr2 = c2.difference(c1)
        cr2 = cr2.intersection(bound_box)
        m_res.append(cr2)
    except:
        print("there is a problem get2anchorsHslRegions cr2")
    return m_res


def getDisjointSubRegions(all_regions_list):
    if len(all_regions_list) == 1:
        return all_regions_list
    head_item = all_regions_list[0]
    return intersect(head_item, getDisjointSubRegions(all_regions_list[1:]))


def fix_precision(shape_, precision_=PRECISION):
    return loads(dumps(loads(shape_.wkt), rounding_precision=precision_))


def intersect(item, disjoint_regions):
    res = []
    cr_itemm = None
    for region in disjoint_regions:
        itemm = fix_shape(item)
        if itemm is not None and not itemm.is_empty:
            regionn = fix_shape(region)
            if regionn is not None and not regionn.is_empty:
                try:
                    common_area = itemm.intersection(regionn)
                except:
                    print("Exception One ------------------")
                common_areaa = fix_shape(common_area)
                if common_areaa is not None and not common_areaa.is_empty:
                    res.append(common_areaa)
                    try:
                        cr_item = itemm.difference(regionn)
                    except:
                        print("Exception Two ---------------")
                    cr_itemm = fix_shape(cr_item)
                    try:
                        cr_region = regionn.difference(itemm)
                    except:
                        print("Exception Three ---------------")
                    cr_regionn = fix_shape(cr_region)
                    if cr_regionn is not None and not cr_regionn.is_empty:
                        res.append(cr_regionn)
                else:  # common_areaa is None
                    # there is not common area between item and region, add region only.
                    # We do not know for other regions, so do not do anything for item
                    res.append(regionn)
            # region is empty do not do anything
        else:
            # item is None
            regionn = fix_shape(region)
            if regionn is not None and not regionn.is_empty:
                res.append(region)
        if cr_itemm is not None and not cr_itemm.is_empty:
            item = cr_itemm
    if cr_itemm is None and not item.is_empty:
        res.append(item)
    else:
        res.append(cr_itemm)
    return res


def fix_shape(shape_):
    if shape_ is None:
        return None  # this is kind of exit(0)
    if shape_.geom_type == 'MultiPolygon':
        poly_list = [fix_precision(x) for x in shape_ if (x.area > 0.0001 and fix_precision(x).is_valid)]
        if len(poly_list) == 0:
            return None  # Polygon()
        elif len(poly_list) == 1:
            if fix_precision(poly_list[0]).is_valid:
                return Polygon(fix_precision(poly_list[0]))
            else:
                return None
        else:
            if fix_precision(MultiPolygon(poly_list)).is_valid:
                return fix_precision(MultiPolygon(poly_list))  # MultiPolygon(poly_list)
            else:
                return None
    elif shape_.geom_type == 'Polygon':
        # here it is a polygon
        if shape_.area > 0.0001 and fix_precision(shape_).is_valid:
            return fix_precision(shape_)
        else:
            return None  # Polygon()
    else:
        return None


def getExpectation(list_):
    if len(list_) == 0:
        return 0
    else:
        sum_ = 0.0
        sum_sqr = 0.0
        for x in list_:
            sum_sqr += x.area * x.area
            sum_ += x.area
        if sum_ == 0:
            return 0
        else:
            return sum_sqr / sum_


def get_entropy(list_):
    if len(list_) == 0:
        return 0
    else:
        area_sum_ = np.sum([x.area for x in list_])
        px_ = [x.area/area_sum_ for x in list_]
        return entropy(px_,base=len(px_)),[x.area for x in list_]


def all_positions(max_x=MAX_X, max_y=MAX_Y, tics=TICS):
    positions = []
    for i in range(max_x // tics + 1):
        for j in range(max_y // tics + 1):
            positions.append((i * tics, j * tics))
    return positions


def color(current, total):
    if current * 100 / total < 50:
        return "\033[91m %d \033[0m" % current
    if current * 100 / total < 75:
        return "\033[93m %d \033[0m" % current
    return "\033[92m %d \033[0m" % current
