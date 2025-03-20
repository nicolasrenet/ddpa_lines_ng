#!/usr/bin/env python3
from PIL import Image, ImagePath
import json
import numpy as np
import skimage as ski
from didip_handwriting_datasets import charters_htr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import itertools
import re
import pytest


if __name__ == '__main__':

    # 1. open image as gray  and line metadata

    img = Image.open(sys.argv[1])
    line_dict = json.load(open(re.sub(r'.img.jpg', '.lines.gt.json', sys.argv[1])))

    img_hwc = np.array( img )
    img_hw = ski.color.rgb2gray( img_hwc )
    img_hw = ski.filters.rank.mean( img_hw, footprint=np.full((10,10),1))
    thresh = ski.filters.threshold_otsu( img_hw )
    binary_img = img_hw < thresh
    img_mask = np.zeros(img_hw.shape, dtype='bool')



    region_props = []

    for line in line_dict['lines']:
        polygon_coordinates = [ tuple(pair) for pair in line['coreBoundary']  ]
        line_bbox = ImagePath.Path( polygon_coordinates ).getbbox()
        base_line = line['baseline']

        leftx, topy = line_bbox[:2]
        polygon_coordinates = np.array( polygon_coordinates, dtype='int')[:,::-1]

        # mean kernel
        boolean_mask = np.array( ski.draw.polygon2mask( img_hw.shape, polygon_coordinates ), dtype='bool')
        img_mask += boolean_mask

    median_line_height = np.median( [ line['strokeWidth'] for line in line_dict['lines'] ])
    binary_img *= img_mask

    labeled_ccs = ski.measure.label( binary_img )
    #img_label_overlay = ski.color.label2rgb( labeled_ccs, img, bg_label=0)

    for region in ski.measure.regionprops( labeled_ccs ):
        if region.area > median_line_height**2/4:
                region_props.append( region.bbox )


    fig, ax = plt.subplots(figsize=(100,300))
    #ax.imshow( img_label_overlay )
    ax.imshow( img )

    regions = sorted( ski.measure.regionprops( labeled_ccs ), key=lambda b: np.mean([b.bbox[1],b.bbox[3]]))
    for r,region in enumerate(regions):
        if region.area > median_line_height**2/4:
            #print(region.bbox)
            miny, minx, maxy, maxx = region.bbox
            ax.add_patch( mpatches.Rectangle( (minx, miny), maxx-minx, maxy-miny, fill=False,edgecolor='red',linewidth=1))
            plt.text(minx, miny, f'{r}')
    print(len(regions), " regions")


    #plt.show()

    # starting: each box is its set
    sets = [ [(s_idx,bb)] for s_idx,bb in enumerate(sorted(regions, key=lambda b: np.mean([b.bbox[1],b.bbox[3]]))) ]
    dilation = 5



    count=0;
    while (count<2):
        print([ [ elt[0] for elt in s ] for s in sets ])
        print("dilation=", dilation)
        updated_sets = []
        for i,s1 in enumerate(sets):
            updated_sets.append( s1 )
            #for s2 in sets[i+1:]:
            for s2 in sets:
                if s1 is s2:
                    continue
                s1_first_y_top, s1_first_x_left, s1_first_y_bottom, s1_first_x_right = s1[0][1].bbox
                s1_last_y_top, s1_last_x_left, s1_last_y_bottom, s1_last_x_right = s1[-1][1].bbox
                s2_first_y_top, s2_first_x_left, s2_first_y_bottom, s2_first_x_right = s2[0][1].bbox
                s2_last_y_top, s2_last_x_left, s2_last_y_bottom, s2_last_x_right = s2[-1][1].bbox

                if s1_last_x_right+dilation >= s2_first_x_left-dilation:
                    print("s1_first_y_top=",s1_first_y_top, "s1_first_y_bottom=",s1_first_y_bottom)
                    print("s2_first_y_top=",s2_first_y_top, "s2_first_y_bottom=",s2_first_y_bottom)
                    if ((s2_first_y_top >= s1_last_y_top and s1_last_y_top >= s2_first_y_bottom) or (s1_last_y_top >= s2_first_y_top and s2_first_y_top >= s1_last_y_bottom)):
                        print(s1[0][0], s1[0][1].bbox, s2[0][0], s2[0][1].bbox)
                        updated_sets.pop()
                        updated_sets.append( s1+s2 )
                        break # each box intersects with at most one R-neighbor
        dilation += 1
        sets = sorted(updated_sets, key=lambda s: np.mean([s[0][1].bbox[1],s[-1][1].bbox[3]])) 
        #print([ [ elt[0] for elt in s ] for s in sets ])
        count+=1
            
                    




def projection_method():

    x_projection = np.sum( binarized_line, axis=0)

    xs = list(itertools.chain.from_iterable([ [x]*nbr for x,nbr in enumerate(x_projection) ]))

    plt.subplot(211)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.hist( xs, np.arange( binarized_line.shape[1]))
    plt.subplot(212)
    plt.imshow( binarized_line )
    #plt.plot( np.arange( binarized_line.shape[1]), x_projection )
    plt.show()

    black_rls = []
    start_current = -1
    for x,projx in enumerate(x_projection):
        # if 0-value while interval not open yet
        if projx > 1 and start_current < 0:
            start_current = x
        elif projx == 0 and start_current >= 0:
            black_rls.append( (start_current, x ))
            start_current = -1

    filtered_rls = [ interv for interv in black_rls if interv[1]-interv[0] >= median_line_height ] 
    print("{} rls total, with {} valid".format( len(black_rls), len(filtered_rls))) 
    actual_x_rls = [ (x1+leftx, x2) for (x1, x2)  in filtered_rls ]
    print(actual_x_rls)


def intersect_x( bbox1, bbox2, dilation=0):
    bbox1_y_top, bbox1_x_left, bbox1_y_bottom, bbox1_x_right = bbox1
    bbox2_y_top, bbox2_x_left, bbox2_y_bottom, bbox2_x_right = bbox2

    if bbox1_x_right >= bbox2_x_left:
        print('x-intersection!')
        if bbox2_y_top >= bbox1_y_top:
            print('y-intersection - first test')
            print("Now testing bbox1_y_top={} >= bbox2_y_bottom={}".format( bbox1_y_top, bbox2_y_bottom))
            if bbox1_y_top >= bbox2_y_bottom:
                return True
        if bbox1_y_top >= bbox2_y_top:
            print('y-intersection - first test')
            print("Now testing bbox2_y_top={} >= bbox1_y_bottom={}".format( bbox2_y_top, bbox1_y_bottom))
            if bbox2_y_top >= bbox1_y_bottom:
                return True
    return False

def test_box_intersect_1():

    # no intersection
                      # y1 x1  y2  x2
    assert intersect_x((5, 10, 15, 30), (20, 40, 25, 50)) == False

    # x-intersection only
    assert intersect_x((5, 10, 15, 30), (20, 28, 25, 50)) == False

    # y-intersection only (case 1)
    assert intersect_x((5, 10, 15, 30), (12, 40, 25, 50)) == False
    # y-intersection only (case 2)
    assert intersect_x((5, 10, 15, 30), (3, 40, 25, 50)) == False

    
    # intersection (case 1)
    print("Intersection test (case 1)")
    assert intersect_x( (5, 10, 15, 30), (12, 28, 25, 50))
    # intersection (case 2)
    assert intersect_x( (5, 10, 15, 30), (3, 28, 25, 50)) 

    
    


