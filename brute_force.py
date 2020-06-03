# -*- coding: utf-8 -*-
import argparse
import time
from our_library import *

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

print("Running brute force on tics=" + str(tics) + ", anchors=" + str(nb_anchors))

# Param

minAvgRA = 9999999999

positions = all_positions(max_x=max_x, max_y=max_y, tics=tics)
anchors_list = list(combinations(positions, nb_anchors))

start = time.time()
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
end = time.time()

drawNetwork(optimal_anchors, optimal_areas, algo_="brute",max_x_=max_x, max_y_=max_y)

print("**Optimal Anchor Pos.:" + str(optimal_anchors), minAvgRA)
print('Running Times : ' + str(round((end - start) / 60.0, 2)) + ' (min.)')

f_res = open('./TXT/brutee.txt', 'a')
f_res.write(str(optimal_anchors)+';'+str(minAvgRA)+';'+str(end - start)+';'+str(nb_anchors)+';'+str(tics)+'\n')
f_res.close()
