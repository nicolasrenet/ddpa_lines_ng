#!/usr/bin/env python3


"""
Convert a segmentation JSON in LTRB format to a YOLO label file:

Input:

```
```

Output:

```
0 0.194099 0.109710 0.018351 0.013629
0 0.261152 0.101193 0.099379 0.027257
0 0.359684 0.102044 0.101073 0.026235
0 0.149351 0.140545 0.041220 0.035775
0 0.234190 0.131175 0.112084 0.024532
0 0.171231 0.173595 0.043196 0.042589
```

"""
import json
import sys
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
import random
import os
import itertools

sys.path.append(os.environ['HOME']+'/graz/htr/vre/ddpa_lines')

from libs import seglib




def polygons_to_yolo_segmentation( polygs: np.ndarray, img_size):
    """
    Polygon convertion:  -> YOLO (minus label) = <x1>, <y1>, <x2>, <y2>, ..., <x_n>, <y_n> (normalized)
    """
    


def plot_polygons( bbs: np.ndarray, img_size: tuple, format='ltrb'):
    """
    Display image + bounding boxes.  Default format is LTRB.

    Args:
        format (str): input format
            + 'ltrb': rectangle coordinates (default)
            + 'coco': normalized, COCO-style bounding box coordinates (center point + size)
            + 
    """
    if format == 'coco':
        bbs = coco_detect_to_ltrb( bbs, img_size)
    elif format == 'yolo':
        bbs = yolo_segmentation_to_ltrb( bbs, img_size)
    fig, ax = plt.subplots(figsize=(200,200))
    for left, top, right, bottom in bbs:
        ax.add_patch( mpatches.Rectangle( (left, top), right-left, bottom-top, fill=False, edgecolor='red', linewidth=1))
    ax.imshow(img)
    plt.show()


def convert_segfile_to_yolo( file_prefix, label=0 ):
    """
    Convert segmentation JSON to normalized, YOLO-style segmentation trainset: x_center, y_center, width, height
    """

    img_path = Path(file_prefix).with_suffix('.img.jpg')
    img = Image.open( img_path, 'r')
    annotation_path = Path(file_prefix).with_suffix('.lines.gt.json')
    output_filename = img_path.with_suffix('.txt')
    label = label

    with open( annotation_path, 'r') as line_dict_file, open(output_filename, 'w') as outf :
        line_dict = json.load( line_dict_file )
        for polygon in [ list(itertools.chain.from_iterable([(x/img.width, y/img.height) for (x,y) in l['coreBoundary']])) for l in line_dict['lines']]:
            outf.write(f"{label} ")
            outf.write(" ".join(["{:.6f}".format( fld ) for fld in polygon]))
            outf.write('\n')


def split_set( *arrays, test_size=.2, random_state =46):
    seq = range(len(arrays[0]))
    train_set = set(random.sample( seq, int(len(arrays[0])*(1-test_size))))
    test_set = set(seq) - train_set
    sets = []
    for a in arrays:
        sets.extend( [[ a[i] for i in train_set ], [ a[j] for j in test_set ]] )
    return sets


if __name__ == '__main__':

    imgs = list(Path('.').glob('*.jpg'))
    labels = [ img.with_suffix('.txt') for img in imgs ]

    # conversion
    for img in imgs:
        convert_segfile_to_yolo( img.stem )
        

    # split sets
    img_train, img_test, lbl_train, lbl_test = split_set( imgs, labels, test_size=.2)
    img_train, img_val, lbl_train, lbl_val = split_set( img_train, lbl_train, test_size=.1)

    img_train_set, lbl_train_set = set( img_train), set(lbl_train)
    assert not img_train_set.intersection( set(img_test) ) and not img_train_set.intersection( set(img_val))
    assert not lbl_train_set.intersection( set(lbl_test) ) and not lbl_train_set.intersection( set(lbl_val))


    for sf in ('images', 'labels'):
        for ss in ('train', 'val', 'test'):
            for p in Path( sf, ss ).glob('*'):
                p.unlink()
    for ss, img_set in (('train', img_train), ('val', img_val), ('test', img_test)):
        for img in img_set:
            shutil.copy( img, 'images/{}/{}'.format( ss, img.with_suffix('.jpg')))
            shutil.copy( img.with_suffix('.txt'), 'labels/{}'.format( ss ))

    



