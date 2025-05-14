import numpy as np
import random
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Union,Callable
import skimage as ski
from torchvision.tv_tensors import BoundingBoxes, Mask




random.seed(46)

def get_n_color_palette(n: int, s=.85, v=.95) -> list:
    """
    Generate n well-distributed random colors. Use golden ratio to generate colors from the HSV color
    space.

    Reference: https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/

    Args:
        n (int): number of color to generate.

    Returns:
        list: a list of (R,G,B) tuples
    """
    golden_ratio_conjugate = 0.618033988749895
    random.seed(13)
    h = random.random()
    palette = np.zeros((1,n,3))
    for i in range(n):
        h += golden_ratio_conjugate
        h %= 1
        palette[0][i]=(h, s, v)
    return (ski.color.hsv2rgb( palette )*255).astype('uint8')[0].tolist()

def display_mask_heatmaps( masks: Tensor ):
    """ Display heapmap for the combined page masks (sum over boxes).
    """
    plt.imshow(torch.sum(masks, axis=0).permute(1,2,0).detach().numpy())
    plt.show()
 
def display_line_masks_raw( preds: list[dict], box_threshold=.8, mask_threshold=.2 ):
    """
    For each page, for each box above the threshold, display the line masks in turn.
    """
    for msks,sc in [ (p['masks'].detach().numpy(),p['scores'].detach().numpy()) for p in preds ]:
        print(len(msks))
        for m in msks[sc>box_threshold]:
            m = m[0]
            plt.imshow( m*(m>mask_threshold) )
            plt.show()


def display_random_predictions(count=2, model_file='best.mlmodel', random_state=46):
    random.seed( random_state )
    img_paths = random.sample( list(Path('dataset').glob('*.jpg')), count)
    imgs_out, preds = predict( img_paths, model_file=model_file)
    maps = [ post_process( p ) for p in preds ]
    vizs = batch_visuals( [ {'img':img_t, 'id':str(path)} for img_t,path in zip(imgs_out, img_paths)], maps, color_count=0 )
    for mp, atts, path in viz:
        plt.imshow(mp)
        plt.title(path)
        for label, centroid, _, _ in atts:
            plt.txt(*centroid[:0:-1], label, size=15)
        plt.show()

def display_dir_predictions(directory, model_file='best.mlmodel'):
    import time
    for img_path in list(Path(directory).glob('*.jpg')):
        start = time.time()
        img_t, out = predict( [img_path], model_file=model_file)
        print("Prediction: {:.5f}s".format( time.time()-start))
        start = time.time()
        mp, atts, path = batch_visuals( [ {'img':img_t[0], 'id':str(img_path)} ], out, color_count=0 )[0]
        print("Visual: {:.5f}s".format( time.time()-start))
        plt.imshow( mp )
        plt.title( path )
        for label, centroid, _, _ in atts:
            plt.txt(*centroid[:0:-1], label, size=15)
        plt.show()

def batch_visuals( inputs:list[Union[Tensor,dict]], raw_maps: list[tuple[np.ndarray,dict]], threshold=.2, color_count=-1, alpha=.4):
    """
    Given a list of image tensors and a list of tuples (<labeled map>,<attributes>), returns page images
    with mask overlays, as well as attributes.

    Args:
        inputs (list[Tensor]): a list of image tensors, or dictionaries with 'img' tensor
        raw_maps (list[tuple[np.ndarray,dict]]): a list of tuples with
            - labeled map 
            - attributes: i.e. dictionary of the form `{'masks': ..., 'boxes': ..., 'scores': ... }`
    Returns:
        list[tuple[np.array, dict, str]]: a list of tuples (img_HWC, attributes, id)
    """
    assert isinstance(inputs[0], Tensor) or ( type(inputs[0]) is dict and 'img' in inputs[0] )

    imgs, ids, maps, attr = [], [], [], []
    if isinstance(inputs[0], Tensor):
        imgs = [ img.cpu().numpy() for img in inputs ] 
        ids = [ f"image-{i}" for i in range(len(imgs)) ]
    elif type(inputs[0]) is dict and 'img' in inputs[0]:
        imgs=[ img['img'].cpu().numpy() for img in inputs ]
        ids = [ img['id'] if 'id' in img else f'image-{i}' for (i,img) in enumerate(inputs) ] 

    default_color = [0,0,1.0] # BLUE
    for img,mp in zip(imgs,raw_maps):
        # generate labeled masks
        labeled_msk, attributes = mp
        labeled_msk = np.transpose( labeled_msk, (1,2,0))
        bm = labeled_msk.astype('bool')
        img = np.transpose( img, (1,2,0))
        img_complementary = img * ( ~bm + bm * (1-alpha))
        col_msk = None
        if color_count>=0:
            colors = get_n_color_palette( color_count ) if color_count > 0 else get_n_color_palette( np.max(labeled_msk))
            col_msk = np.zeros( img.shape, dtype=img.dtype )
            for l in range(1, np.max(labeled_msk)+1):
                col = np.array(colors[l % len(colors) ])
                col_msk += (labeled_msk==l) * (col/255.0)
            col_msk *= alpha
        # single color
        else:
            # BLUE * BOOL * ALPHA
            col_msk = np.full(img.shape, default_color) * bm * alpha
        composed_img_array = img_complementary + col_msk
        # Combination: (H,W,C), i.e. fit for image viewers and plots
        maps.append(composed_img_array)
        attr.append(attributes)
    
    return list(zip(maps, attr, ids))


def display_annotated_img( img: Tensor, target: dict, alpha=.4, color='g'):
    """ Overlay of instance masks.
    Args:
        img (Tensor): (C,H,W) image
        target (dict[str,Tensor]): a dictionary of labels with
        - 'masks'=(N,H,W) tensor of masks, where N=# instances for image
        - 'boxes'=(N,4) tensor of BB coordinates
        - 'labels'=(N) tensor of box labels
    """
    img = img.detach().numpy()
    masks = target['masks'].detach().numpy()
    masks = [ m * (m>.5) for m in masks ]
    boxes = [ [ int(c) for c in box ] for box in target['boxes'].detach().numpy().tolist()]
    bm = np.sum( masks, axis=0).astype('bool')
    col = {'r': [1.0,0,0], 'g':[0,1.0,0], 'b':[0,0,1.0]}[color]
    # RED * BOOL * ALPHA
    red_mask = np.transpose( np.full((img.shape[2],img.shape[1],3), col), (2,0,1)) * bm * alpha
    img_complementary = img * ( ~bm + bm * (1-alpha))
    composed_img_array = np.transpose(img_complementary + red_mask, (1,2,0))
    pil_img = Image.fromarray( (composed_img_array*255).astype('uint8'))
    draw = ImageDraw.Draw( pil_img )
    polygon_boundaries = [[ [box[0],box[0]], [box[0],box[1]], [box[1],box[1]], [box[1],box[0]] ] for box in boxes] 
    for i,polyg in enumerate(polygon_boundaries):
        if i%2 != 0:
            draw.polygon(polyg, outline='blue')
    plt.imshow( np.array( pil_img ))
    plt.show()


def gt_masks_to_labeled_map( masks: Mask ):
    """
    Combine stacks of GT line masks (as in data annotations) into a single, labeled page-wide map.
    """
    return np.sum( np.stack([ m * lbl for (lbl,m) in enumerate(masks, start=1)]), axis=0)

