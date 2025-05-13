import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.append( str(Path(__file__).parents[1]))

import line_detect_train as ld

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
    print(ld.simple_metrics( map_gt, map_pred, iou_threshold=0.22, foreground=None))
    print(ld.simple_metrics( map_gt, map_pred, iou_threshold=0.22, foreground=map_foreground))

