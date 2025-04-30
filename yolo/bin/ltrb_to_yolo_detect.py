#!/usr/bin/env python3


"""
Convert a segmentation JSON in LTRB format to a YOLO label file:

Input:

```
"rectangles_ltrb":[
    [272,381,306,414],
    [308,352,423,437],
    [393,370,532,441],
    [541,360,681,401],
    [700,366,719,390]
    ...]
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
import random



def ltrb_to_coco_detect( bbs: np.ndarray, img_size):
    """
    Bounding box conversion for detection: LTRB -> COCO (minus label) = <centerX>, <centerY>, <width>, <height> (normalized)
    """
    center_x_width_norm = np.array(list(zip( (bbs[:,0]+bbs[:,2])/2, bbs[:,2]-bbs[:,0]))) / img_size[0]
    center_y_height_norm = np.array(list(zip( (bbs[:,1]+bbs[:,3])/2, bbs[:,3]-bbs[:,1]))) / img_size[1]
    return np.array(list(zip( center_x_width_norm[:,0], center_y_height_norm[:,0], center_x_width_norm[:,1], center_y_height_norm[:,1] )))


def ltrb_to_yolo_segmentation( bbs: np.ndarray, img_size):
    """
    Bounding box conversion: LTRB -> YOLO (minus label) = <x1>, <y1>, <x2>, <y2>, ..., <x_n>, <y_n> (normalized)
    """
    xs_norm = bbs[:,[0,2]] / img_size[0]
    ys_norm = bbs[:,[1,3]] / img_size[1]
    return np.hstack([ xs_norm[:,0,None], ys_norm[:,0,None], 
                        xs_norm[:,0,None], ys_norm[:,1,None],
                        xs_norm[:,1,None], ys_norm[:,1,None],
                        xs_norm[:,1,None], ys_norm[:,0,None]])

def coco_detect_to_ltrb( bbs: np.ndarray, img_size ):
    """
    Bounding box conversion: COCO -> LTRB 
    """
    widths, heights = bbs[:,2]*img_size[0], bbs[:,3]*img_size[1]
    center_xs, center_ys = bbs[:,0]*img_size[0], bbs[:,1]*img_size[1]
    lefts, rights = center_xs-widths/2, center_xs+widths/2
    tops, bottoms = center_ys-heights/2, center_ys+heights/2
    return np.array(list(zip(lefts, tops, rights, bottoms )))


def yolo_segmentation_to_ltrb( bbs: np.ndarray, img_size ):
    """
    Bounding box conversion: YOLO -> LTRB 
    """
    print((bbs[:,[0,4]] * img_size[0])[:5])
    print((bbs[:,[1,5]] * img_size[1])[:5])
    return np.hstack( [ bbs[:,0,None] * img_size[0], bbs[:,1,None] * img_size[1], bbs[:,4,None] * img_size[0], bbs[:,5,None] * img_size[1]])



def plot_boxes( bbs: np.ndarray, img_size: tuple, format='ltrb'):
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


def convert_ltrb_file_to_yolo( file_prefix, label=0 ):
    """
    Convert LTRB to normalized, YOLO-style segmentation trainset: x_center, y_center, width, height
    """

    img_path = Path(file_prefix).with_suffix('.jp2')
    annotation_path = Path(file_prefix).with_suffix('.gt.json')
    label = label

    annotation = json.load( open(annotation_path, 'r'))
    img = Image.open( img_path, 'r')
    print( img_path )

    img_width, img_height = img.size
    bboxes = np.array( annotation['rectangles_ltrb'] )

    #bboxes_norm = ltrb_to_yolo( bboxes, img.size )
    bboxes_norm = np.hstack( [ np.full((len(bboxes),1), label), ltrb_to_coco_detect( bboxes, img.size ) ])
    #print("Raw boxes:", bboxes[:5])
    #print("YOLO bboxes:", bboxes_norm[:5])
    #print("YOLO->Raw:", yolo_to_ltrb(bboxes_norm[:,1:], img.size)[:5])

    output_filename = img_path.with_suffix('.txt')
    with open(output_filename, 'w') as output_file:
        for bb in bboxes_norm:
            output_file.write("{:d} ".format(int(bb[0])))
            output_file.write(" ".join([ "{:.6f}".format( fld ) for fld in bb[1:] ]))
            output_file.write("\n")

    #plot_boxes( bboxes_norm[:,1:], img.size, format='yolo' )


def split_set( *arrays, test_size=.2, random_state =46):
    seq = range(len(arrays[0]))
    train_set = set(random.sample( seq, int(len(arrays[0])*(1-test_size))))
    test_set = set(seq) - train_set
    sets = []
    for a in arrays:
        sets.extend( [[ a[i] for i in train_set ], [ a[j] for j in test_set ]] )
    return sets


if __name__ == '__main__':

    imgs = list(Path('.').glob('*.jp2'))
    labels = [ img.with_suffix('.txt') for img in imgs ]

    # conversion

    for img in imgs:
        convert_ltrb_file_to_yolo( img.stem )

    # split sets
    img_train, img_test, lbl_train, lbl_test = split_set( imgs, labels, test_size=.2)
    img_train, img_val, lbl_train, lbl_val = split_set( img_train, lbl_train, test_size=.1)

    img_train_set, lbl_train_set = set( img_train), set(lbl_train)
    assert not img_train_set.intersection( set(img_test) ) and not img_train_set.intersection( set(img_val))
    assert not lbl_train_set.intersection( set(lbl_test) ) and not lbl_train_set.intersection( set(lbl_val))


    for ss in ('train', 'val', 'test'):
        for sf in ('images', 'labels'):
            for p in Path( sf, ss ).glob('*'):
                p.unlink()
    for ss, img_set in (('train', img_train), ('val', img_val), ('test', img_test)):
        for img in img_set:
            shutil.copy( img, 'images/{}/{}'.format( ss, img.with_suffix('.jpg')))
            shutil.copy( img.with_suffix('.txt'), 'labels/{}'.format( ss ))

    



