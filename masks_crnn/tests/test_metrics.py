import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.append( str(Path(__file__).parents[1]))

from libs import seglib 

@pytest.fixture(scope="module")
def data_path():
    return Path(__file__).parent.joinpath('data')

def test_metrics( data_path ):
    # GT map
#    ds=ld.ChartersDataset( list(Path('./hard_cases').glob('*.jpg')), list(Path('./hard_cases').glob('*.json')), img_size=1024)
#    map_gt=ld.gt_masks_to_labeled_map(ds[2][1]['masks']) # merging all binary masks into a flat, labeled one 
#
#    # Predicted map
#    imgs = list( Path('hard_cases').glob('*.jpg') )
#    imgs, preds = ld.predict( imgs, model_file='./best.mlmodel')
#    map_pred = np.squeeze(ld.post_process( preds[2] )[0])
    map_gt, map_pred, map_foreground = [ np.load(data_path.joinpath(filename)) for filename in ('map_gt.npy', 'map_pred.npy', 'binary_img.npy') ]
    #print(metrics.simple_metrics( map_gt, map_pred, iou_threshold=0.22, foreground=None))
    #print(metrics.simple_metrics( map_gt, map_pred, iou_threshold=0.22, foreground=map_foreground))
    pixel_metrics = seglib.polygon_pixel_metrics_two_flat_maps( map_gt, map_pred)
    print( seglib.polygon_pixel_metrics_to_pixel_based_scores( pixel_metrics ))
    print( seglib.polygon_pixel_metrics_to_line_based_scores( pixel_metrics, threshold=.8 ))


def test_mAP( data_path ):

    import line_detect_train as ld, random
    ds=ld.ChartersDataset( list(Path('./hard_cases').glob('*.jpg'))[:5], list(Path('./hard_cases').glob('*.json'))[:5])
    gt_maps = [ segviz.gt_masks_to_labeled_map(ds[i][1]['masks']) for i in range(5) ]
    imgs, preds = ld.predict( list(Path('./hard_cases').glob('*.jpg'))[:5], model_file='./best.mlmodel');
    pred_maps = [ np.squeeze(ld.post_process( p, mask_threshold=.2, box_threshold=.5 )[0]) for p in preds ]
    pms = [ seglib.polygon_pixel_metrics_two_flat_maps( m1, m2 ) for m1, m2 in zip( pred_maps, gt_maps) ]
    print(seglib.mAP( pms ))


