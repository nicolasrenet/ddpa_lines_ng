
from pathlib import Path
import time
import sys
import matplotlib.pyplot as plt
import random

sys.path.append( str(Path(__file__).parents[1] ))
import ddp_lineseg as lsg
from libs import segviz

import fargv

p = {
    'model_file': 'best.mlmodel',
    'mask_threshold': 0.25,
    'rescale': 0,
    'img_paths': set([]),
    'directory': '.',
    'color_count': (0, "-1 for single color, n > 1 for fixed number of colors, 0 for 1 color/line"),
    'limit': (5, "How many file to display."),
    'random': 0,
}


if __name__ == '__main__':

    args, _ = fargv.fargv(p)

    live_model = lsg.SegModel.load( args.model_file )

    files = []
    if args.img_paths:
        files = [ Path(p) for p in args.img_paths ]
    else:
        files = random.sample(list(Path(args.directory).glob('*.jpg')), args.limit) if random else list(Path(args.directory).glob('*.jpg'))[:args.limit]


    for img_path in files:
        print(img_path)
        #start = time.time()
        imgs_t, preds, sizes = lsg.predict( [img_path], live_model=live_model)

        #print("Prediction: {:.5f}s".format( time.time()-start))

        maps = []
        #start = time.time()
        if args.rescale:
            maps=[ lsg.post_process( p, orig_size=sz, mask_threshold=args.mask_threshold ) for (p,sz) in zip(preds,sizes) ]
            mp, atts, path = segviz.batch_visuals( [img_path], maps, color_count=0 )[0]
        else:
            maps=[ lsg.post_process( p, mask_threshold=args.mask_threshold ) for p in preds ]
            mp, atts, path = segviz.batch_visuals( [ {'img':imgs_t[0], 'id':str(img_path)} ], maps, color_count=0 )[0]
        #print("Visual: {:.5f}s".format( time.time()-start))

        plt.imshow( mp )
        plt.title( path )
        for att_dict in atts:
            label, centroid = att_dict['label'], att_dict['centroid']
            plt.text(*centroid[:0:-1], label, size=15)
        plt.show()

