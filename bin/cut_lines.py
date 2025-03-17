#!/usr/bin/env python3
from PIL import Image, ImagePath
import json
import numpy as np
import skimage as ski
from didip_handwriting_datasets import charters_htr
import matplotlib.pyplot as plt
import sys



# 1. open image as gray  and line metadata

page_img = Image.open('data/f1ea2db678e04f513d2238604c73839f.Wr_OldText.0.img.jpg')
line_dict = json.load(open('data/f1ea2db678e04f513d2238604c73839f.Wr_OldText.0.lines.gt.json'))

polygon_coordinates = [ tuple(pair) for pair in line_dict['lines'][5]['coreBoundary']  ]
line_bbox = ImagePath.Path( polygon_coordinates ).getbbox()




# 2. Extract bounding box, construct a binary mask for the polygon, add median background

bbox_img = page_img.crop( line_bbox )
bbox_img_hwc = np.array( bbox_img )
bbox_img_hw = ski.color.rgb2gray( bbox_img_hwc )
#plt.imshow( bbox_img_hw )
#plt.show()
leftx, topy = line_bbox[:2]
transposed_coordinates = np.array([ (x-leftx, y-topy) for x,y in polygon_coordinates ], dtype='int')[:,::-1]

# mean kernel
bbox_img_hw = ski.filters.rank.mean( bbox_img_hw, footprint=np.full((10,10),1))

thresh = ski.filters.threshold_otsu( bbox_img_hw )
binary_bbox = bbox_img_hw < thresh


boolean_mask = ski.draw.polygon2mask( bbox_img_hwc.shape[:2], transposed_coordinates )

plt.imshow( binary_bbox * boolean_mask )
#plt.imshow( binary_bbox )
plt.show()


# 3. Otsu

# 4. 

# 3. Use vertical 
