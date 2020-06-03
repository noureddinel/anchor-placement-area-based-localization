import argparse
import time
from our_library import *
import itertools
from copy import deepcopy

shapely.speedups.enable()

parser = argparse.ArgumentParser()
parser.add_argument('--max_x', help='Give the MAX_X value', default=MAX_X, type=int)
parser.add_argument('--max_y', help='Give the MAX_Y value', default=MAX_Y, type=int)
parser.add_argument('--points', help='Give the TICS value', default=NB_POINT_PER_SIDE, type=int)
parser.add_argument('--nb_anchors', help='Give the number of anchors', default=NB_ANCHORS, type=int)

args = parser.parse_args()

max_x = args.max_x
max_y = args.max_y
n_points = args.points
tics = max_x // (n_points - 1)
nb_anchors = args.nb_anchors

print("Running heuristic on tics=" + str(tics) + ", anchors=" + str(nb_anchors))


def neighbor(point, target_tics_):
    """ Function return the neighbors of a point x in the matrix A
    take as input the coordinate of x and dim the dimension of A then
    returns the neighbors of x in the matrix B which have 2*dim elements.
    exemple:
    neighbor([2,1],4)
    returns : [[3, 1], [3, 2], [3, 3], [4, 1], [4, 2], [4, 3], [5, 1], [5, 2], [5, 3]]
    """
    a = np.array(point)
    adjacents = []
    for i in range(a[0] - target_tics_, a[0] + target_tics_ + 1, target_tics_):
        if 0 <= i <= max_x:
            for j in range(a[1] - target_tics_, a[1] + target_tics_ + 1, target_tics_):
                if 0 <= j <= max_y:
                    adjacents.append([i, j])
    return adjacents


def get_neighbor_list(initial_list, target_tics):
    anchors_list = []
    neighbor_list = [neighbor(x, target_tics_=target_tics) for x in initial_list]
    for items in itertools.product(*neighbor_list):
        anchors_list.append(list(items))
    return anchors_list


# Param
# For every number of anchors, we need to initialise the initial vector which gives the optimal
# anchor placement for the low discretisation.

initial_list = [[(0, 96), (96, 96), (96, 192)],
                [(0, 96), (96, 0), (96, 96), (192, 96)],
                [(0, 0), (0, 96), (0, 192), (96, 96), (192, 96)]] #we get this from brute force


initial = initial_list[nb_anchors-3]# chose the right initial according to nb_anchors
all_points =[3,5,9,17,33,97]
all_tics = [max_x // (p - 1) for p in all_points]
list_tics = [t for t in all_tics if t>=tics][1:]


start = time.time()
for _tics in list_tics:
    anchors_list = get_neighbor_list(initial, target_tics=_tics)
    minAvgRA = 999999999
    optimal_anchors = []
    for index, anchors in enumerate(tqdm(anchors_list)):
        l = getAllSubRegions(anchors_=anchors, max_x_=max_x, max_y_=max_y)
        res = getDisjointSubRegions(l)
        avgRA = getExpectation(res)
        if avgRA != 0:
            if minAvgRA > avgRA:
                minAvgRA = avgRA
                optimal_anchors = []
                for a in anchors:
                    optimal_anchors.append(a)
                optimal_areas = res
    initial = deepcopy(optimal_anchors)
end = time.time()

drawNetwork(optimal_anchors, optimal_areas, algo_="fear_heuristic")

print("**Optimal Anchor Pos.:" + str(optimal_anchors), minAvgRA)
print('Runinig Times : ' + str(round((end - start) / 60.0, 2)) + ' (min.)')

f_res = open('./TXT/heuristic.txt', 'a')
f_res.write(str(optimal_anchors)+';'+str(minAvgRA)+';'+str(end - start)+';'+str(nb_anchors)+';'+str(tics)+'\n')

f_res.close()
