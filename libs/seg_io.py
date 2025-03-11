from pathlib import Path
import random
from typing import Tuple
import json

import numpy as np
from PIL import Image, ImageDraw
import skimage as ski
import torch
from torch import Tensor
import matplotlib.pyplot as plt

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from . import seglib


def display_polygon_lines_from_img_and_xml_files( img_file: str, page_xml: str, color_count=2) -> np.ndarray:
    """
    Render a single set of polygons, as lines.

    Args:
        img_file (str): path to the original manuscript image.
        page_xml (str): path to the segmentation metadata (PageXML)
        color_count (int): number of colors in the palette. If 0 (default), use as many colors as polygons.
    Returns:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    with Image.open(img_file) as input_img_hw:

        colors = get_n_color_palette( color_count ) if color_count else get_n_color_palette(int( polygon_count ))
        colors = [ tuple(c) for c in colors ]
        draw = ImageDraw.Draw( input_img_hw )
        segmentation_dict = seglib.segmentation_dict_from_xml( page_xml )
        polygon_boundaries = [ [ tuple(xy) for xy in line['boundary']] for line in segmentation_dict['lines']]
        for p, polyg in enumerate(polygon_boundaries, start=1):
            draw.polygon( polyg, fill=colors[ p%len(colors) ], width=5 )
        return np.array( input_img_hw )
            
def display_polygon_lines_from_img_and_json_files( img_file: str, seg_json: str, color_count=2) -> np.ndarray:
    """
    Render a single set of polygons, as lines.

    Args:
        img_file (str): path to the original manuscript image.
        seg_json (str): path to the segmentation metadata (JSON)
        color_count (int): number of colors in the palette. If 0 (default), use as many colors as polygons.
    Returns:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    with Image.open(img_file) as input_img_hw, open(seg_json) as seg_json_file:

        colors = get_n_color_palette( color_count ) if color_count else get_n_color_palette(int( polygon_count ))
        colors = [ tuple(c) for c in colors ]
        draw = ImageDraw.Draw( input_img_hw )
        segmentation_dict = json.load( seg_json_file )
        polygon_boundaries = [ [ tuple(xy) for xy in line['boundary']] for line in segmentation_dict['lines']]
        for p, polyg in enumerate(polygon_boundaries, start=1):
            draw.polygon( polyg, fill=colors[ p%len(colors) ], width=5 )
        return np.array( input_img_hw )
            

def display_polygon_lines_from_img_and_dict( img_file: str, segdict: dict, color_count=2) -> np.ndarray:
    """
    Render a single set of polygons, as lines.

    Args:
        img_file (str): path to the original manuscript image.
        segdict (str): segmentation metadata (a JSON dictionary)
        color_count (int): number of colors in the palette. If 0 (default), use as many colors as polygons.
    Returns:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    with Image.open(img_file) as input_img_hw:

        print(input_img_hw)
        colors = get_n_color_palette( color_count ) if color_count else get_n_color_palette(int( polygon_count ))
        colors = [ tuple(c) for c in colors ]
        draw = ImageDraw.Draw( input_img_hw )
        polygon_boundaries = [ [ tuple(xy) for xy in line['boundary']] for line in segdict['lines']]
        print(polygon_boundaries)
        for p, polyg in enumerate(polygon_boundaries, start=1):
            print("draw_line()", polygon_boundaries)
            draw.polygon( polyg, fill=colors[ p%len(colors) ], width=5 )
        return np.array( input_img_hw )
            


def display_polygon_map_from_img_and_xml_files( img_file: str, page_xml: str, color_count=0, alpha=.75) -> np.ndarray:
    """
    Render a single set of polygons, as a map.

    Args:
        img_file (str): path to the original manuscript image.
        page_xml (str): path to the pickled polygon set, encoded as a 4-channel, 8-bit tensor.
        color_count (int): number of colors in the palette. If 0 (default), use as many colors as polygons.
    Returns:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    with Image.open(img_file) as input_img_hw:
        polygons_chw = seglib.polygon_map_from_xml_file( page_xml ) 
        return display_polygon_set( input_img_hw, polygons_chw, color_count, alpha )

def display_polygon_map_from_img_and_json_files( img_file: str, seg_json: str, color_count=0, alpha=.75) -> np.ndarray:
    """
    Render a single set of polygons, as a map.

    Args:
        img_file (str): path to the original manuscript image.
        seg_json (str): path to the pickled polygon set, encoded as a 4-channel, 8-bit tensor.
        color_count (int): number of colors in the palette. If 0 (default), use as many colors as polygons.
    Returns:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    with Image.open(img_file) as input_img_hw:
        polygons_chw = seglib.polygon_map_from_json_file( seg_json )
        return display_polygon_set( input_img_hw, polygons_chw, color_count, alpha )

def display_polygon_map_from_img_and_polygon_map( img_file: str, polygons_chw: Tensor, color_count=0, alpha=.75) -> np.ndarray:
    """
    Render a single set of polygons using two colors (alternate between odd- and even-numbered lines).

    Args:
        img_file (str): path to the original manuscript image.
        polygons_chw (Tensor): polygon set, encoded as a 4-channel, 8-bit tensor.
        color_count (int): number of colors in the palette. If 0 (default), use as many colors as polygons.
    Returns:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    with Image.open(img_file) as input_img_hw:
        return display_polygon_set( input_img_hw, polygons_chw, color_count, alpha )

def display_two_polygon_sets_from_img_and_tensor_files( img_file: str, polygon_file_1: str, polygon_file_2: str, bg_alpha=.75) -> np.ndarray:
    """
    Render two sets of polygons (typically: GT and pred.) using two colors, for human diagnosis.
   
    Args:
        img_file (str): path to the original manuscript image.
        polygon_file_1 (str): path to the first pickled polygon set, encoded as a 4-channel, 8-bit tensor.
        polygon_file_2 (str): path to the second pickled polygon set, encoded as a 4-channel, 8-bit tensor.
    Returns:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    with Image.open(img_file) as input_img_hw:
        polygons_1_chw = torch.load( polygon_file_1 )
        polygons_2_chw = torch.load( polygon_file_2 )
        return display_two_polygon_sets( input_img_hw, polygons_1_chw, polygons_2_chw, bg_alpha )

def display_polygon_set( input_img_hw: Image.Image, polygons_chw: Tensor, color_count=0, alpha=.75 ) -> np.ndarray:
    """
    Render a single set of polygons.

    Args:
        input_img_hw (Image.Image): the original manuscript image, as opened with PIL.
        polygons_chw (Tensor): polygon set, encoded as a 4-channel, 8-bit tensor.
        color_count (int): number of colors in the palette. If 0 (default), use as many colors as polygons.
       
    Returns:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """

    input_img_hwc = np.asarray( input_img_hw )
    if tuple(polygons_chw.shape[1:]) != input_img_hwc.shape[:2]:
        raise ValueError("Polygon map and input image have different height/length: respectively {} and {}".format( tuple(polygons_chw.shape[1:]), input_img_hwc.shape[:2]) +
                "\nCheck that the source for the map (an XML segmentation file or a dictionary) correctly describes the image size.")
    polygon_count = torch.max( polygons_chw )

    colors = get_n_color_palette( color_count ) if color_count else get_n_color_palette(int( polygon_count ))

    fg_masked_hwc = np.zeros( input_img_hwc.shape ) 

    output_img = input_img_hwc.copy()
    
    for p in range(1, polygon_count+1 ):
        # flat binary mask
        polygon_mask_hw = seglib.mask_from_polygon_map_functional( polygons_chw, lambda m: m == p )
        fg_masked_hwc[ polygon_mask_hw ] += colors[ p % len(colors) ]

    # in original image, transparency applies only to the polygon pixels
    alpha_mask = fg_masked_hwc != 0
    alphas = np.full( alpha_mask.shape, 1.0 )
    alphas[ alpha_mask ] = alpha

    # combine: BG + FG
    # use this statement instead to make the polygons more visible
    #output_img = (input_img_hwc * alpha ) + fg_masked_hwc * (1-alpha)
    output_img = (input_img_hwc * alphas ) + fg_masked_hwc * (1-alpha)

    return output_img.astype('uint8')


def display_two_polygon_sets( input_img_hw: Image.Image, polygons_1_chw: Tensor, polygons_2_chw: Tensor, bg_alpha=.5 ) -> np.ndarray:
    """
    Render two sets of polygons (typically: GT and pred.) using two colors, for human diagnosis.

    Args:
        input_img (Image.Image): the original manuscript image, as opened with PIL.
        polygons_1_chw (Tensor): polygon set #1, encoded as a 4-channel, 8-bit tensor.
        polygons_2_chw (Tensor): polygon set #2, encoded as a 4-channel, 8-bit tensor.
    Returns:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    input_img_hwc = np.asarray( input_img_hw )
    polygon_count_1 = torch.max( polygons_1_chw )
    polygon_count_2 = torch.max( polygons_2_chw )

    #colors = (255,0,0), (0,0,255)
    colors = get_n_color_palette( 2, s=.99, v=.99 )

    fg_masked_hwc = np.zeros( input_img_hwc.shape ) 

    output_img = input_img_hwc.copy()
    
    # create a single mask for each set
    mask_1_hw = seglib.mask_from_polygon_map_functional( polygons_1_chw, lambda m: m != 0 )
    mask_2_hw = seglib.mask_from_polygon_map_functional( polygons_2_chw, lambda m: m != 0 )

    fg_masked_hwc[ mask_1_hw ] = colors[0]
    fg_masked_hwc[ mask_2_hw ] += colors[1]

    # in original image, transparency applies only to the polygon pixels
    alpha_mask = fg_masked_hwc != 0
    alphas = np.full( alpha_mask.shape, 1.0 )
    alphas[ alpha_mask ] = bg_alpha

    # combine: BG + FG
    output_img = (input_img_hwc * alphas ) + fg_masked_hwc * (1-bg_alpha)
    
    return output_img.astype('uint8')

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

 


def display_boxes_and_masks(imgs, row_title=None, **imshow_kwargs):
    """
    Display masks and boxes, as overlay.

    Args:
        imgs (List[Tuple[Tensor,Dict[str,Tensor]]]): a sequence of pairs with
            + input image (a tensor)
            + a dictionary with 'boxes' as (N,4)-tensors and 'masks' as (N,H,W)-binary tensors.
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
