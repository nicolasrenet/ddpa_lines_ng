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




# 1. open image as gray  and line metadata

page_img = Image.open('./data/boxes/f1ea2db678e04f513d2238604c73839f.Wr_OldText.0.img.jpg')
line_dict = json.load(open('./data/boxes/f1ea2db678e04f513d2238604c73839f.Wr_OldText.0.lines.gt.json'))

region_props = []

for line in line_dict['lines']:
    polygon_coordinates = [ tuple(pair) for pair in line['coreBoundary']  ]
    line_bbox = ImagePath.Path( polygon_coordinates ).getbbox()
    base_line = line['baseline']
    line_height = line['strokeWidth']


    bbox_img = page_img.crop( line_bbox )
    bbox_img_hwc = np.array( bbox_img )
    bbox_img_hw = ski.color.rgb2gray( bbox_img_hwc )
    leftx, topy = line_bbox[:2]
    transposed_coordinates = np.array([ (x-leftx, y-topy) for x,y in polygon_coordinates ], dtype='int')[:,::-1]

    # mean kernel
    bbox_img_hw = ski.filters.rank.mean( bbox_img_hw, footprint=np.full((10,10),1))

    thresh = ski.filters.threshold_otsu( bbox_img_hw )
    binary_bbox = bbox_img_hw < thresh
    boolean_mask = ski.draw.polygon2mask( bbox_img_hwc.shape[:2], transposed_coordinates )
    binarized_line = binary_bbox*boolean_mask

    labeled_ccs = ski.measure.label( binarized_line )
    #img_label_overlay = ski.color.label2rgb( labeled_ccs, bbox_img_hwc, bg_label=0)

    or region in ski.measure.regionprops( labeled_ccs ):
        if region.area > line_height**2/4:
            region_props.append( region.bbox )


fig, ax = plt.subplots(figsize=(400,800))
#ax.imshow( img_label_overlay )
ax.imshow( page_img )

    for region in ski.measure.regionprops( labeled_ccs ):
        if region.area > median_line_height**2/4:
            print(region.bbox)
            miny, minx, maxy, maxx = region.bbox
            ax.add_patch( mpatches.Rectangle( (minx, miny), maxx-minx, maxy-miny, fill=False,edgecolor='red',linewidth=2))


plt.show()

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

# cut the baseline according to the invervals above


# (optional) compute a new polygon


# compute the bbox of every segment


        

    


