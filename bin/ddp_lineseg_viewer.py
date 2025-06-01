#!/usr/bin/env python3

# nprenet@gmail.com
# 05.2025

"""
A simple line segmenter/viewer that predicts lines on images and displays the result.

For proper segmentation and recording of a region-based segmentation (crops), see `ddp_line_detect.py`.'
"""

# stdlib
from pathlib import Path
import time
import sys
import random
import logging

# 3rd party
import matplotlib.pyplot as plt

# DiDip
import fargv

# local
sys.path.append( str(Path(__file__).parents[1] ))
import ddp_lineseg as lsg
from libs import segviz




logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)
# tone down unwanted logging
logging.getLogger('matplotlib.font_manager').disabled=True
logging.getLogger('PIL').setLevel(logging.INFO)


p = {
    'model_path': str(src_root.joinpath("best.mlmodel")),
    'mask_threshold': [0.25, "Threshold used for line masks--a tweak on the post-processing phase."],
    'rescale': [0, "If True, display segmentation on original image; otherwise (default), get the image size from the model used for inference (ex. 1024 x 1024)."],
    'img_paths': set(Path('dataset').glob('*.jpg')),
    'color_count': [0, "Number of colors for polygon overlay: -1 for single color, n > 1 for fixed number of colors, 0 for 1 color/line."],
    'limit': [0, "How many files to display."],
    'random': [0, "If non-null, randomly pick <random> paths out of the <img_paths> list."],
}


if __name__ == '__main__':

    args, _ = fargv.fargv(p)
    logger.debug( args )

    live_model = lsg.SegModel.load( args.model_path )

    files = []
    if args.random:
        files = random.sample([ Path(p) for p in args.img_paths ], args.random)
    else:
        files = [ Path(p) for p in ( list(args.img_paths)[:args.limit] if args.limit else args.img_paths) ]

    for img_path in files:
        logger.info(img_path)
        start = time.time()
        imgs_t, preds, sizes = lsg.predict( [img_path], live_model=live_model)

        logger.debug("Inference time: {:.5f}s".format( time.time()-start))

        maps = []
        start = time.time()
        if args.rescale:
            maps=[ lsg.post_process( p, orig_size=sz, mask_threshold=args.mask_threshold ) for (p,sz) in zip(preds,sizes) ]
            mp, atts, path = segviz.batch_visuals( [img_path], maps, color_count=0 )[0]
        else:
            maps=[ lsg.post_process( p, mask_threshold=args.mask_threshold ) for p in preds ]
            mp, atts, path = segviz.batch_visuals( [ {'img':imgs_t[0], 'id':str(img_path)} ], maps, color_count=0 )[0]
        logger.debug("Rendering time: {:.5f}s".format( time.time()-start))

        plt.imshow( mp )
        plt.title( path )
        for att_dict in atts:
            label, centroid = att_dict['label'], att_dict['centroid']
            plt.text(*centroid[:0:-1], label, size=15)
        plt.show()

