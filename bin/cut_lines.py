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
    print(img_mask.dtype)

median_line_height = np.median( [ line['strokeWidth'] for line in line_dict['lines'] ])

print(binary_img.dtype)

binary_img *= img_mask

labeled_ccs = ski.measure.label( binary_img )
#img_label_overlay = ski.color.label2rgb( labeled_ccs, img, bg_label=0)

for region in ski.measure.regionprops( labeled_ccs ):
    if region.area > median_line_height**2/4:
            region_props.append( region.bbox )


fig, ax = plt.subplots(figsize=(100,300))
#ax.imshow( img_label_overlay )
ax.imshow( img )

regions = ski.measure.regionprops( labeled_ccs )
for region in regions:
    if region.area > median_line_height**2/4:
        #print(region.bbox)
        miny, minx, maxy, maxx = region.bbox
        ax.add_patch( mpatches.Rectangle( (minx, miny), maxx-minx, maxy-miny, fill=False,edgecolor='red',linewidth=1))
print(len(regions), " regions")



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


        

    


