
#stdlib
from pathlib import Path
import json
from typing import List, Tuple, Callable, Optional, Dict, Union, Mapping, Any
import itertools
import re
import copy

# 3rd-party
from PIL import Image, ImageDraw
import skimage as ski
import xml.etree.ElementTree as ET
import torch
from torch import Tensor
import numpy as np
import numpy.ma as ma


__LABEL_SIZE__=8

"""Functions for segmentation output management

+ storing polygons on tensors (from dictionaries or pageXML outputs)
+ computing IoU and F1 scores over GT/predicted label maps

A note about types:

+ PageXML or JSON: initial input (typically: from segmentation framework)
+ torch.Tensor: map storage and computations (eg. counting intersections)
+ np.ndarray: metrics and scores; initial mapping of labels: use 32-bit _signed_ integers
  for storing compound (intersecting) labels, to ensure smooth conversion into
  tensors.
"""


def polygon_map_from_json_file(  segmentation_json: str) -> Tensor:
    """Read line polygons from a JSON file and store them into a tensor, as pixel maps.
    Channels allow for easy storage of overlapping polygons.

    Args:
        segmentation_json (str): path of a JSON file

    Returns:
        Tensor: the polygons rendered as a 4-channel image (a tensor).
    """
    with open( segmentation_json, 'r' ) as json_file:
        return polygon_map_from_segmentation_dict( json.load( json_file ))


def polygon_map_from_xml_file( page_xml: str ) -> Tensor:
    """Read line polygons from a PageXML file and store them into a tensor, as pixel maps.
    Channels allow for easy storage of overlapping polygons.

    Args:
        page_xml (str): path of a PageXML file.

    Returns:
        Tensor: the polygons rendered as a 4-channel image (a tensor).
    """

    segmentation_dict = segmentation_dict_from_xml( page_xml )
    return polygon_map_from_segmentation_dict( segmentation_dict)

def polygon_map_from_segmentation_dict( segmentation_dict: dict ) -> Tensor:
    """Store line polygons into a tensor, as pixel maps.

    Args:
        segmentation_dict (dict): kraken's segmentation output, i.e. a dictionary of the form::

                 {
                 'image_wh': [w, h],
                 'text_direction': '$dir',
                 'type': 'baseline',
                 'lines': [
                   {'baseline': [[x0, y0], [x1, y1], ...], 'boundary': [[x0, y0], [x1, y1], ... [x_m, y_m]]},
                   ...
                 ]
                 'regions': [ ... ] }

    Returns:
        Tensor: the polygons rendered as a 4-channel image.
    """
    polygon_boundaries = [ line['boundary'] for line in segmentation_dict['lines'] ]

    # create 2D matrix of 32-bit integers
    # (fillPoly() only accepts signed integers - risk of overflow is non-existent)
    mask_size = segmentation_dict['image_wh'][::-1]

    label_map = np.zeros( mask_size, dtype='int32' )

    # rendering polygons
    for lbl, polyg in enumerate( polygon_boundaries ):
        points = np.array(polyg)[:,::-1] # x <-> y
        polyg_mask = ski.draw.polygon2mask( mask_size, points )
        apply_polygon_mask_to_map( label_map, polyg_mask, lbl+1 )

    #Image.fromarray( label_map ).show()

    # 8-bit/pixel, 4 channels (note: order is little-endian)
    polygon_img = array_to_rgba_uint8( label_map )
    #ski.io.imshow( polygon_img.permute(1,2,0).numpy() )

    return polygon_img


def line_binary_mask_from_json_file(segmentation_json: str ) -> Tensor:
    """From a segmentation dictionary describing polygons, return a boolean mask where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        segmentation_json (str): a JSON file describing the lines.

    Returns:
        Tensor: a flat boolean tensor with size (H,W)
    """
    with open( segmentation_json, 'r' ) as json_file:
        return line_binary_mask_from_segmentation_dict( json.load( json_file ))

def line_binary_mask_from_xml_file( page_xml: str ) -> Tensor:
    """From a PageXML file describing polygons, return a boolean mask where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        page_xml (str): a Page XML file describing the lines.

    Returns:
        Tensor: a flat boolean tensor with size (H,W)
    """
    segmentation_dict = segmentation_dict_from_xml( page_xml )
    return line_binary_mask_from_segmentation_dict( segmentation_dict )


def line_binary_mask_from_segmentation_dict( segmentation_dict: dict ) -> Tensor:
    """From a segmentation dictionary describing polygons, return a boolean mask where any pixel belonging
    to a polygon is 1 and the other pixels 0.
    Note: masks can be built from a polygon map and an arbitrary selection function, with
    the mask_from_polygon_map_functional() function below.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed
            from a JSON file.

    Returns:
        Tensor: a flat boolean tensor with size (H,W)
    """

    polygon_boundaries = [ line['boundary'] for line in segmentation_dict['lines'] ]

    # create 2D boolean matrix
    mask_size = segmentation_dict['image_wh'][::-1]
    page_mask_hw = np.zeros( mask_size, dtype='bool')

    # rendering polygons
    for lbl, polyg in enumerate( polygon_boundaries ):
        points = np.array( polyg )[:,::-1]
        polyg_mask = ski.draw.polygon2mask( mask_size, points )
        page_mask_hw += polyg_mask

    return torch.tensor( page_mask_hw )


def line_images_from_img_xml_files(img: str, page_xml: str ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation PageXML file describing polygons, return
    a list of pairs (<line cropped BB>, <polygon mask>).

    Args:
        img (str): the input image's file path
        page_xml: :type page_xml: str a Page XML file describing the
            lines.

    Returns:
        list: a list of pairs (<line image BB>: np.ndarray (HWC), mask:
        np.ndarray (HW))
    """
    with Image.open(img, 'r') as img_wh:
        segmentation_dict = segmentation_dict_from_xml( page_xml )
        return line_images_from_img_segmentation_dict( img_wh, segmentation_dict )


def line_images_from_img_json_files( img: str, segmentation_json: str ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation JSON file describing polygons, return
    a list of pairs (<line cropped BB>, <polygon mask>).

    Args:
        img (str): the input image's file path
        segmentation_json (str): path of a JSON file

    Returns:
        list: a list of pairs (<line image BB>: np.ndarray (HWC), mask: np.ndarray (HW))
    """
    with Image.open(img, 'r') as img_wh, open( segmentation_json, 'r' ) as json_file:
        return line_images_from_img_segmentation_dict( img_wh, json.load( json_file ))

def line_images_from_img_segmentation_dict(img_whc: Image.Image, segmentation_dict: dict ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """From a segmentation dictionary describing polygons, return 
    a list of pairs (<line cropped BB>, <polygon mask>).

    Args:
        img_whc (Image.Image): the input image (needed for the size information).
        segmentation_dict: :type segmentation_dict: dict a dictionary, typically constructed from a JSON file.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: a list of pairs (<line
        image BB>: np.ndarray (HWC), mask: np.ndarray (HWC))
    """
    polygon_boundaries = [ line['boundary'] for line in segmentation_dict['lines'] ]
    img_hwc = np.asarray( img_whc )

    pairs_line_bb_and_mask = []# [None] * len(polygon_boundaries)

    for lbl, polyg in enumerate( polygon_boundaries ):

        points = np.array( polyg )[:,::-1] # polygon's points ( x <-> y )
        page_polyg_mask = ski.draw.polygon2mask( img_hwc.shape, points ) # np.ndarray (H,W,C)
        y_min, x_min, y_max, x_max = np.min( points[:,0] ), np.min( points[:,1] ), np.max( points[:,0] ), np.max( points[:,1] )
        line_bbox = img_hwc[y_min:y_max+1, x_min:x_max+1] # crop both img and mask
        # note: mask has as many channels as the original image
        bb_label_mask_hwc = page_polyg_mask[y_min:y_max+1, x_min:x_max+1]

        #pairs_line_bb_and_mask[lbl]=( line_bbox, bb_label_mask )
        pairs_line_bb_and_mask.append( (line_bbox, bb_label_mask_hwc) )

    return pairs_line_bb_and_mask

def line_images_from_img_polygon_map(img_wh: Image.Image, polygon_map_chw: Tensor) -> List[Tuple[np.ndarray, np.ndarray]]:
    """From a tensor storing polygons, return a list of pairs (<line cropped BB>, <polygon mask>).

    Args:
        img_whc (Image.Image): the input image (needed for the size information).
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: a list of pairs (<line image BB>: np.ndarray (HWC), mask: np.ndarray (HW))
    """

    max_label = torch.max( polygon_map_chw )
    img_hwc = np.array( img_wh )

    pairs_line_bb_and_mask = []# [None] * max_label

    for lbl in range(1, max_label+1 ):
        page_label_mask_hw = retrieve_polygon_mask_from_map( polygon_map_chw, lbl )

        # BB of non-zero pixels
        non_zero_ys, non_zero_xs = page_label_mask_hw.numpy().nonzero()
        y_min, x_min, y_max, x_max = np.min(non_zero_ys), np.min(non_zero_xs), np.max(non_zero_ys), np.max(non_zero_xs)
        line_bbox = img_hwc[y_min:y_max+1, x_min:x_max+1]

        bb_label_mask = expand_flat_tensor_to_n_channels(page_label_mask_hw[y_min:y_max+1, x_min:x_max+1], 3)

        #pairs_line_bb_and_mask[ lbl-1 ]=(line_bbox, bb_label_mask) 
        pairs_line_bb_and_mask.append( (line_bbox, bb_label_mask) )

    return pairs_line_bb_and_mask



def line_masks_from_img_xml_files(img: str, page_xml: str ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation PageXML file describing polygons, return
    the bounding box coordinates and the boolean masks.

    Args:
        img (str): the input image's file path
        page_xml (page_xml): str a Page XML file describing the lines.

    Returns:
        Tuple[np.ndarray,np.ndarray]: a pair of tensors: a tensor (N,4) of BB coordinates tuples,
            and a tensor (N,H,W) of page-wide line masks.
    """
    with Image.open(img, 'r') as img_wh:
        segmentation_dict = segmentation_dict_from_xml( page_xml )
        return line_masks_from_img_segmentation_dict( img_wh, segmentation_dict )


def line_masks_from_img_json_files( img: str, segmentation_json: str, key='boundary' ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation JSON file describing polygons, return
    the bounding box coordinates and the boolean masks.

    Args:
        img (str): the input image's file path
        segmentation_json (str): path of a JSON file

    Returns:
        Tuple[np.ndarray,np.ndarray]: a pair of tensors: a tensor (N,4) of BB coordinates tuples,
            and a tensor (N,H,W) of page-wide line masks.
    """
    with Image.open(img, 'r') as img_wh, open( segmentation_json, 'r' ) as json_file:
        return line_masks_from_img_segmentation_dict( img_wh, json.load( json_file ), key=key)

def line_masks_from_img_segmentation_dict(img_whc: Image.Image, segmentation_dict: dict, key='boundary' ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """From a segmentation dictionary describing polygons, return 
    the bounding box coordinates and the boolean masks.

    Args:
        img_whc (Image.Image): the input image (needed for the size information).
        segmentation_dict: :type segmentation_dict: dict a dictionary, typically constructed from a JSON file.

    Returns:
        Tuple[np.ndarray,np.ndarray]: a pair of tensors: a tensor (N,4) of BB coordinates tuples,
            and a tensor (N,H,W) of page-wide line masks.
    """
    polygon_boundaries = [ line[key] for line in segmentation_dict['lines'] ]
    img_hwc = np.asarray( img_whc )

    bbs = []
    masks = []

    for polyg in polygon_boundaries:
        points = np.array( polyg )[:,::-1]  # polygon's points ( x <-> y )
        page_polyg_mask = ski.util.img_as_ubyte(ski.draw.polygon2mask( img_hwc.shape[:2], points )) # np.ndarray (H,W)
        y_min, x_min, y_max, x_max = [ float(p) for p in (np.min( points[:,0] ), np.min( points[:,1] ), np.max( points[:,0] ), np.max( points[:,1] )) ]
        bbs.append( (x_min,y_min,x_max,y_max) )
        masks.append( page_polyg_mask )

    return (np.stack( bbs ), np.stack( masks ))



def expand_flat_tensor_to_n_channels( t_hw: Tensor, n: int ) -> np.ndarray:
    """Expand a flat map by duplicating its only channel into n identical ones.
    Channels dimension is last for convenient use with PIL images.

    Args:
        t_hw (Tensor): a flat map.
        n (int): number of (identical) channels in the resulting tensor.

    Returns:
        np.ndarray: a (H,W,n) array.
    """
    if len(t_hw.shape) != 2:
        raise TypeError("Function expects a 2D map!")
    t_hwc = t_hw.reshape( t_hw.shape+(1,)).expand(-1,-1,n)
    return t_hwc.numpy()

def segmentation_dict_from_xml(page: str) -> Dict[str,Union[str,List[Any]]]:
    """Given a pageXML file name, return a JSON dictionary describing the lines.

    Args:
        page (str): path of a PageXML file

    Returns:
        Dict[str,Union[str,List[Any]]]: a dictionary of the form::

            {"text_direction": ..., "type": "baselines", "lines": [{"tags": ..., "baseline": [ ... ]}]}

    """
    direction = {'0.0': 'horizontal-lr', '0.1': 'horizontal-rl', '1.0': 'vertical-td', '1.1': 'vertical-bu'}

    page_dict: Dict[str, Union['str', List[Any]]] = { 'type': 'baselines', 'text_direction': 'horizontal-lr' }

    def construct_line_entry(line: ET.Element, regions: list = [] ) -> dict:
            #print(regions)
            line_id = line.get('id')
            baseline_elt = line.find('./pc:Baseline', ns)
            if baseline_elt is None:
                return None
            bl_points = baseline_elt.get('points')
            if bl_points is None:
                return None
            baseline_points = [ [ int(p) for p in pt.split(',') ] for pt in bl_points.split(' ') ]
            coord_elt = line.find('./pc:Coords', ns)
            if coord_elt is None:
                return None
            c_points = coord_elt.get('points')
            if c_points is None:
                return None
            polygon_points = [ [ int(p) for p in pt.split(',') ] for pt in c_points.split(' ') ]

            return {'line_id': line_id, 'baseline': baseline_points, 
                    'boundary': polygon_points, 'regions': regions} 

    def process_region( region: ET.Element, line_accum: list, regions:list ):
        regions = regions + [ region.get('id') ]
        for elt in list(region.iter())[1:]:
            if elt.tag == "{{{}}}TextLine".format(ns['pc']):
                line_entry = construct_line_entry( elt, regions )
                if line_entry is not None:
                    line_accum.append( construct_line_entry( elt, regions ))
            elif elt.tag == "{{{}}}TextRegion".format(ns['pc']):
                process_region(elt, line_accum, regions)

    with open( page, 'r' ) as page_file:

        # extract namespace
        ns = {}
        for line in page_file:
            m = re.match(r'\s*<([^:]+:)?PcGts\s+xmlns(:[^=]+)?=[\'"]([^"]+)["\']', line)
            if m:
                ns['pc'] = m.group(3)
                page_file.seek(0)
                break

        if 'pc' not in ns:
            raise ValueError(f"Could not find a name space in file {page}. Parsing aborted.")

        lines = []

        page_tree = ET.parse( page_file )
        page_root = page_tree.getroot()

        pageElement = page_root.find('./pc:Page', ns)
        
        page_dict['imagename']=pageElement.get('imageFilename')
        page_dict['image_wh']=[ int(pageElement.get('imageWidth')), int(pageElement.get('imageHeight'))]
        
        for textRegionElement in pageElement.findall('./pc:TextRegion', ns):
            process_region( textRegionElement, lines, [] )

        page_dict['lines'] = lines

    return page_dict 

def merge_seals_regseg_lineseg( regseg: dict, region_labels: List[str], *linesegs: List[str]) -> dict:
    """Merge 2 segmentation outputs into a single one:

    * the page-wide yolo/seals segmentation (with OldText, ... regions)
    * the line segmentation for the regions defined in the first one

    The resulting file is a page-wide line-segmentation JSON.

    Args:
        regseg (dict): the regional segmentation json, as given by the 'seals' app
        *linesegs (List[str]): a number of local line segmentations for the region defined in the
            first file, of the form::

                {"type": "baseline", "imagename": ..., lines: [ {"id": "... }, ... ] }

        region_labels (List[str]): in the region segmentation, labels of those regions
            that have been line-segmented.

    Returns:
        dict: a page-wide line-segmentation dictionary.
    """
    charter_img_suffix = '.img.jpg'

    print( region_labels)
    def translate( dictionary, translation ):
        #print("Translate by ", translation)
        #print("Input dictionary has", len(dictionary["lines"]), "lines")
        new_lines = []
        for line in dictionary['lines']:
            new_line = copy.deepcopy( line )
            #print("Before:", line['baseline'])
            for k in ('baseline', 'boundary'):
                new_line[k] = [ [int(x+translation[0]),int(y+translation[1])] for (x,y) in line[k]]
            new_lines.append( new_line )
            #print("After:", new_line['baseline'])
        #print("translated", len( new_lines ))

        #print("Input dictionary has", len(dictionary["lines"]), "lines")
        return new_lines

    # extract mapping region type id -> region name
    clsid_2_clsname = { i:n for (i,n) in enumerate( regseg['class_names'] )}
    to_keep = [ i for (i,v) in enumerate( regseg['rect_classes'] ) if clsid_2_clsname[v] in region_labels ]

    # assumptions: line segs are passed in the same order as the region order in the regseg
    to_keep = to_keep[:len(linesegs)]

    # go through local line segmentations (and corresponding rectangle in regseg),
    # and translate every x,y coordinates by value of the rectangle's origin (left,top)
    lines = []
    #img_name = str(Path(linesegs[0]['imagename']).parents[1].joinpath( regseg['img_md5'] ).with_suffix( charter_img_suffix ))
    img_name = "TODO (ANGULAS)"

    for (lineseg, coords) in zip( linesegs, [ c for (index, c) in enumerate( regseg['rect_LTRB'] ) if index in to_keep ]):
        lines.extend( translate( lineseg, translation=coords[:2] ))
        #print("merged lines have now", len(lines))

    merged_seg = { "type": "baselines", 
                   "imagename": img_name,
                   "text_direction": 'horizontal-lr',
                   "script_detection": False,
                   "lines": lines,
                 }

    return merged_seg
        
def seals_regseg_to_crops( img: Image.Image, regseg: dict, region_labels: List[str] ) -> Tuple[List[Image.Image], List[str]]:
    """From a seals-app segmentation dictionary, returns the regions with matching
    labels as a list of images.

    Args:
        img (Image.Image): Image to crop.
        regseg (dict): the regional segmentation json, as given by the 'seals' app
        region_labels (List[str]): Labels to be extracted.

    Returns:
        Tuple[List[Image.Image], List[str]]: a pair with a list of images (H,W)> and a list of class names.
    """

    clsid_2_clsname = { i:n for (i,n) in enumerate( regseg['class_names'] )}
    to_keep = [ i for (i,v) in enumerate( regseg['rect_classes'] ) if clsid_2_clsname[v] in region_labels ]

    return ( [ img.crop( regseg['rect_LTRB'][i] ) for i in to_keep ],
             [ clsid_2_clsname[ regseg['rect_classes'][i]]  for i in to_keep ] )


def seals_regseg_check_class(regseg: dict, region_labels: List[str] ) -> List[bool]:
    """From a seals-app segmentation dictionary, check if rectangle with given labels
    have been detected.

    Args:
        regseg (dict): the regional segmentation json, as given by the 'seals' app
        region_labels (List[str]): Labels to check.

    Returns:
        List[bool]: a list of boolean values.
    """

    clsname_2_clsid = { n:i for (i,n) in enumerate( regseg['class_names'] )}
    
    output = None
    try:
        output = [ clsname_2_clsid[l] in regseg['rect_classes'] for l in region_labels ]
    except KeyError as e:
        print(f"Class label {e} does not exist in the segmentation file.")
    return output




def apply_polygon_mask_to_map(label_map: np.ndarray, polygon_mask: np.ndarray, label: int) -> None:
    """In the segmentation map, label pixels matching a given polygon. Up to 4 labels
    can be stored on a single pixel. A label cannot be applied twice to the same
    map.

    Args:
        label_map (np.ndarray): the map that stores the polygons
        polygon_mask (np.ndarray): a binary mask representing the polygon to be labeled.
        label (int): label for this polygon; if the pixel already has a previous label l', the resulting,
            compound value is `(l'<<8)+label`. Eg. 1. Applying label 4 on a pixel that
            already stores label 2 yields `2 << 8 + 4 = 0x204 = 8192`
            Eg. 2. Pixel ``0x10403`` stores labels ``[1, 4, 3]``

    """
    label_limit = 0xff
    max_three_polygon_label = 0xffffff

    # a label may not use more than 1 byte.
    # With large label values, 4-polygon intersections may result in negative compound label values, though,
    # which is not an issue (the map is meant to be stored and used as an cube of unsigned bytes).
    if label > label_limit:
        raise OverflowError('Overflow: label value ({}) exceeds the limit ({}).'.format( label, label_limit ))

    # Handling duplicated labels:
    if array_has_label(label_map, label):
        raise ValueError("The label map already contains a label with value ({})".format(label))

    # for every pixel in intersection...
    intersection_boolean_mask = np.logical_and( label_map, polygon_mask )
    # if intersection does not already contain a 3-polygon pixel
    if np.any( label_map[ intersection_boolean_mask ] > max_three_polygon_label ):
        maxed_out_pixels = np.transpose(((label_map * intersection_boolean_mask) > max_three_polygon_label).nonzero())
        raise ValueError('Cannot store more than 4 polygons on the same pixel! Following positions maxed out: {}{}'.format(
            repr([ (row,col) for (row,col) in maxed_out_pixels ][:5]),
            ' ...' if len(maxed_out_pixels)>5 else ''))
    # ... shift it
    label_map[ intersection_boolean_mask ] <<= 8

    # only then add label to all pixels matching the polygon
    label_map += polygon_mask.astype( label_map.dtype ) * label


def array_to_rgba_uint8( img_hw: np.ndarray ) -> Tensor:
    """Converts a numpy array of 32-bit integers into a 4-channel tensor.

    Args:
        img_hw (np.ndarray): a flat label map of 32-bit integers.

    Returns:
        Tensor: a 4-channel (c,h,w) tensor of unsigned 8-bit integers.
    """
    if len(img_hw.shape) != 2:
        raise TypeError(format("Input map should have shape (W,H) (actual: {}).".format( img_hw.shape )))
    if img_hw.dtype != 'int32': 
        raise TypeError("Label map's dtype should 'int32' (actual: {}".format( img_hw.dtype ))
    img_hw_32b = img_hw.astype('int32')
    img_chw = torch.from_numpy( np.moveaxis( img_hw_32b.view(np.uint8).reshape( (img_hw.shape[0], -1, 4)), 2, 0))
    return img_chw


def polygon_pixel_metrics_from_img_segmentation_dict(img_whc: Image.Image, segmentation_dict_pred: dict, segmentation_dict_gt: dict, binary_mask: Optional[Tensor]=None) -> np.ndarray:
    """Compute a IoU matrix from an image and two dictionaries describing the segmentation's output (line polygons).

    Args:
        img_whc (Image.Image): the input page, needed for the binarization mask.
        segmentation_dict_pred (dict): a dictionary, typically
            constructed from a JSON file.
        segmentation_dict_gt (dict): a dictionary, typically constructed
            from a JSON file.

    Returns:
        np.ndarray: a 2D array, representing IoU values for each possible pair of polygons.
    """
    #polygon_img_gt: Tensor, polygon_img_pred: Tensor, mask: Tensor) -> Tensor:
    polygon_chw_pred, polygon_chw_gt = [ polygon_map_from_segmentation_dict( d ) for d in (segmentation_dict_pred, segmentation_dict_gt) ]

    binary_mask = get_mask( img_whc )

    return polygon_pixel_metrics_from_polygon_maps_and_mask( polygon_chw_pred, polygon_chw_gt, binary_mask )


def get_mask( img_whc: Image.Image, thresholding_alg: Callable=ski.filters.threshold_otsu ) -> Tensor:
    """Compute a binary mask from an image, using the given thresholding algorithm: FG=1s, BG=0s

    Args:
        img (PIL image): input image

    Returns:
        Tensor: a binary map with FG pixels=1 and BG=0.
    """
    img_hwc= np.array( img_whc )
    threshold = thresholding_alg( ski.color.rgb2gray( img_hwc ) if img_hwc.shape[2]>1 else img_hwc )*255
    img_bin_hw = torch.tensor( (img_hwc < threshold)[:,:,0], dtype=torch.bool )

    return img_bin_hw


def polygon_pixel_metrics_from_polygon_maps_and_mask(polygon_chw_pred: Tensor, polygon_chw_gt: Tensor, binary_hw_mask: Optional[Tensor]=None, label_distance=0) -> np.ndarray:
    """Compute pixel-based metrics from two tensors that each encode (potentially overlapping) polygons
    and a FG mask.

    Args:
        polygon_chw_gt (Tensor): a 4-channel image, where each position may store up
            to 3 overlapping labels (one for each channel)
        polygon_chw_pred (Tensor): a 4-channel image, where each position may store up
            to 3 overlapping labels (one for each channel)
        binary_hw_mask (Tensor): a boolean mask that selects the input image's FG pixel

    Returns:
        np.ndarray: metrics (intersection, union, precision, recall, f1) values for each
            possible pair of labels (i,j) with i ∈  map1 and j ∈ map2. Shared pixels in
            each map (i.e. overlapping polygons) have their weight decreased according to the
            number of polygons they vote for.
    """

    if binary_hw_mask is None:
        binary_hw_mask = torch.full( polygon_chw_gt.shape[1:], 1, dtype=torch.bool )
    if binary_hw_mask.shape != polygon_chw_gt.shape[1:]:
        print("mask shape=", binary_hw_mask.shape, "polygon_chw_gt.shape =", polygon_chw_gt.shape)
        raise TypeError("Wrong type: binary mask should have shape {}".format(polygon_chw_gt.shape[1:]))

    if len(polygon_chw_gt.shape) != 3 or polygon_chw_gt.shape[0] != 4 or polygon_chw_gt.dtype is not torch.uint8:
        raise TypeError("Wrong type: polygon GT map should be a 4-channel tensor of unsigned 8-bit integers.")
    if len(polygon_chw_pred.shape) != 3 or polygon_chw_pred.shape[0] != 4 or polygon_chw_pred.dtype is not torch.uint8:
        raise TypeError("Wrong type: polygon predicted map should be a 4-channel tensor of unsigned 8-bit integers.")
    if polygon_chw_gt.shape != polygon_chw_pred.shape:
        raise TypeError("Wrong type: both maps should have the same shape (instead: {} and {}).".format( polygon_chw_gt.shape, polygon_chw_pred.shape ))

    polygon_chw_fg_pred, polygon_chw_fg_gt = [ polygon_img * binary_hw_mask for polygon_img in (polygon_chw_pred, polygon_chw_gt) ]
    
    metrics = polygon_pixel_metrics_two_maps( polygon_chw_fg_pred, polygon_chw_fg_gt, label_distance )

    return metrics

def retrieve_polygon_mask_from_map( label_map_chw: Tensor, label: int) -> Tensor:
    """From a label map (that may have compound pixels representing polygon intersections),
    compute a binary mask that covers _all_ pixels for the label, whether they belong to an
    intersection or not.

    Args:
        label_map_chw (Tensor): a 4-channel tensor, where each pixel can
            store up to 4 labels.
        label (int): the label to be selected.

    Returns:
        Tensor: a flat, boolean mask for the polygon of choice.
    """
    if len(label_map_chw.shape) != 3 and label_map_chw.shape[0] != 4:
        raise TypeError("Wrong type: label map should be a 4-channel tensor (shape={} instead).".format( label_map_chw.shape ))
    polygon_mask_hw = torch.sum( label_map_chw==label, dim=0).type(torch.bool)

    return polygon_mask_hw


def array_has_label( label_map_hw: np.ndarray, label: int ) -> bool:
    """From a flat label map (as generated from a segmentation dictionary) where each pixel can store up to 3 values,
    test whether a given polygon has been stored already.

    Args:
        label_map_hw (np.ndarray): a 2D map, where each 32-bit integer store up to 4 labels.
        label (int): the label to be checked for.

    Returns:
        bool: True if map already stores the given label; False otherwise.
    """
    if len(label_map_hw.shape) > 2:
        raise TypeError("Map should be a flat map of integers.")

    label_cube_chw = np.moveaxis(label_map_hw.view('uint8').reshape(label_map_hw.shape+(-1,)), 2, 0)
    return bool(np.any( label_cube_chw == label ))



def polygon_pixel_metrics_two_maps( map_chw_1: Tensor, map_chw_2: Tensor, label_distance=5) -> np.ndarray:
    """Provided two label maps that each encode (potentially overlapping) polygons, compute
    for each possible pair of labels (i_pred, j_gt) with i ∈  map1 and j ∈  map2.
    + intersection and union counts
    + precision and recall

    Shared pixels in each map (i.e. overlapping polygons) have their weight decreased according
    to the number of polygons they vote for.

    Args:
        map_chw_1 (Tensor): the predicted map, i.e. a 4-channel map with labeled polygons, with potential overlaps.
        map_hw_2 (Tensor): the GT map, i.e. a 4-channel map with labeled polygons, with potential overlaps.

    Returns:
        np.ndarray: a 4 channel array, where each cell [i,j] stores respectively intersection and union
            counts, as well as precision and recall for a pair of labels [i,j].
    """
    min_label_1, max_label_1 = int(torch.min( map_chw_1[ map_chw_1 > 0 ] ).item()), int(torch.max( map_chw_1 ).item())
    min_label_2, max_label_2 = int(torch.min( map_chw_2[ map_chw_2 > 0 ] ).item()), int(torch.max( map_chw_2 ).item())
    label2index_1 = { l:i for i,l in enumerate( range(min_label_1, max_label_1+1)) }
    label2index_2 = { l:i for i,l in enumerate( range(min_label_2, max_label_2+1)) }

    # 4 channels for the intersection and union counts, and the precision and recall scores, respectively
    metrics_hwc = np.zeros(( max_label_1-min_label_1+1, max_label_2-min_label_2+1, 4), dtype='float32')

    label_range_1, label_range_2 = range(min_label_1, max_label_1+1), range(min_label_2, max_label_2+1)
    
    # Factor out some computations
    label_matrices_1={ l:retrieve_polygon_mask_from_map( map_chw_1, l) for l in label_range_1 }
    label_matrices_2={ l:retrieve_polygon_mask_from_map( map_chw_2, l) for l in label_range_2 }
    depth_1, depth_2 = map_to_depth( map_chw_1 ), map_to_depth( map_chw_2 )
    max_depth = torch.max( depth_1, depth_2 ) # for each pixel, keep the largest depth value of the two maps
    label_counts_1={ l:torch.sum(label_matrices_1[l]/depth_1).item() for l in label_range_1 }
    label_counts_2={ l:torch.sum(label_matrices_2[l]/depth_2).item() for l in label_range_2 }

    for lbl1, lbl2 in itertools.product(label_range_1, label_range_2):
        # assume that labels beyond a given distance do not intersect
        if label_distance > 0 and abs(lbl1-lbl2) > label_distance:
            metrics_hwc[label2index_1[lbl1], label2index_2[lbl2]]=[ 0, label_counts_1[lbl1] + label_counts_2[lbl2], 0, 0 ]
            continue
        # Idea:
        # + the intersection of a label 1 of depth m (where m = # of polygons that intersect on the pixel) with label 2
        # of depth n has weight 1/max(m, n)
        # + the union of a label-1 pixel of depth m with label-2 pixel of depth n has weight (1/m + 1/n)

        # 1. Compute intersection boolean matrix
        #print("1. Compute intersection boolean matrix, depth1 and depth2")
        label_1_matrix, label_2_matrix = label_matrices_1[ lbl1 ], label_matrices_2[ lbl2 ]
        intersection_mask = label_1_matrix * label_2_matrix

        # 2. Compute the weighted intersection count of the two maps
        #print("3. For each pixel, keep the largest depth value of the two maps")
        intersection_count = torch.sum( intersection_mask / max_depth ).item()

        # 3. Compute cardinalities |label 1| and |label 2| in map 1 and map 2, respectively
        #print('4. Compute cardinalities |label 1| and |label 2| in map 1 and map 2, respectively')
        label_1_count, label_2_count = label_counts_1[lbl1], label_counts_2[lbl2]

        # 4. union = |label 1| + |label 2| - |label 1 ∩ label 2|
        #print('5. union = |label 1| + |label 2| - |label 1 ∩ label 2|')
        union_count = label_1_count + label_2_count - intersection_count

        # 5. P = |label 1 ∩ label 2| / | label 2 |; R = |label 1 ∩ label 2| / | label 1 |
        #print('6. P = |label 1 ∩ label 2| / | label 2 |; R = |label 1 ∩ label 2| / | label 1 |')
        if label_1_count != 0:
            # rows (label_1) assumed to be predictions
            precision = intersection_count / label_1_count
        if label_2_count != 0:
            # cols (label_2) assumed to be GT
            recall = intersection_count / label_2_count

        metrics_hwc[label2index_1[lbl1], label2index_2[lbl2]]=[ intersection_count, union_count, precision, recall ]

    return metrics_hwc


def map_to_depth(map_chw: Tensor) -> Tensor:
    """Compute depth of each pixel in the input map, i.e. how many polygons intersect on this pixel.
    Note: 0-valued pixels have depth 1.

    Args:
        map_chw (Tensor): the input tensor (4 channels)

    Returns:
        Tensor: a tensor of integers, where each value represents the
        number of intersecting polygons for the same position in the
        input map.
    """
    depth_map = torch.sum( map_chw != 0, dim=0)
    depth_map[ depth_map == 0 ]=1

    return depth_map


def polygon_pixel_metrics_to_line_based_scores( metrics: np.ndarray, threshold: float=.5 ) -> Tuple[float, float, float, float, float]:
    """Implement ICDAR 2017 evaluation metrics, as described in
    https://github.com/DIVA-DIA/DIVA_Line_Segmentation_Evaluator/releases/tag/v1.0.0
    (a Java implementation)

    IoU = TP / (TP+FP+FN)
    F1 = (2*TP) / (2*TP+FP+FN)

    + find all polygon pairs that have a non-empty intersection
    + sort the pairs by IoU
    + traverse the IoU-descending sorted list and select the first available match
      for each polygon belonging to the prediction set (thus ensuring that no
      polygon can be matched twice)
    + a pred/GT match is TP if both Recall and Precision > .75.
    + a pred/GT match is FP if P < .75; FN if R < .75 (it can be both!)
    + Line-based, per-page IoU [or Jaccard index]= TP/(TP+FP+FN)
    + Document-wide, pixel-based IoU_px  = TP_px/TP_px + FP_px + FN_px}.

    Args:
        metrics (np.ndarray): metrics matrix, with indices [0..m-1, 0..n-1] for labels 1..m, 
            where m and n are the max. labels of of the predicted and GT maps respectively.
            In the channels: intersection count, union count, precision, recall.

    Returns:
        tuple: a 5-tuple with the TP-, FP-, and FN-counts, as well as the Jaccard (aka. IoU)
            and F1 score at the line level.
    """
    label_count_pred, label_count_gt = metrics.shape[:2]

    # find all rows with non-empty intersection
    possible_match_indices = metrics[:,:,0].nonzero()
    match_rows_cols, possible_matches = np.transpose(possible_match_indices), metrics[ possible_match_indices ]
    ious = possible_matches[:,0]/possible_matches[:,1]
    structured_row_col_match_iou = np.array([
        (match_rows_cols[i,0],
         match_rows_cols[i,1],
         possible_matches[i,0],
         possible_matches[i,1],
         possible_matches[i,2],
         possible_matches[i,3],
         ious[i]) for i in range(len(possible_matches))],
         dtype=[('pred_polygon', 'int32'),
                ('gt_polygon', 'int32'),
                ('intersection', 'float32'),
                ('union', 'float32'),
                ('precision', 'float32'),
                ('recall', 'float32'),
                ('iou', 'float32')])

    TP, FP, FN = 0.0, 0.0, 0.0
    
    # sort candidate matches by ascending label order and descending IoU order
    pred_label_iou_copy = structured_row_col_match_iou[['pred_polygon', 'iou']].copy()
    pred_label_iou_copy['iou'] *= -1
    I = np.argsort( pred_label_iou_copy, order=['pred_polygon', 'iou'])

    # select one-to-one matches
    pred2match = { i:False for i in possible_match_indices[0] }
    for possible_match in structured_row_col_match_iou[I]:
        # ensure that each predicted label is matched to at most one GT label
        # (first hit is the one with highest IoU)
        if not pred2match[possible_match['pred_polygon']]:
            pred2match[possible_match['pred_polygon']]=True
            precision, recall = possible_match[['precision', 'recall']]
            TP += (precision >= threshold and recall >= threshold )
            # a FP is a non-zero (Pred, GT) pair whose P < .75 or: the system detects
            # a polygon that partially capture the GT, but too much of the rest also
            FP += precision < threshold
            # a FN is a non-zero (Pred, GT) pair whose R < .75 or: the system detects
            # a polygon that matches the GT, but not enough of it
            FN += recall < threshold

    Jaccard = TP / (TP+FP+FN)
    F1 = 2*TP / (2*TP+FP+FN)

    return (TP, FP, FN, Jaccard, F1)


def polygon_pixel_metrics_to_pixel_based_scores( metrics: np.ndarray) -> Tuple[float, float]:
    """Implement ICDAR 2017 pixel-based evaluation metrics, as described in
    Simistira et al., ICDAR2017, "Competition on Layout Analysis for Challenging Medieval
    Manuscripts", 2017.

    Two versions of the pixel-based IoU metric:
    + Pixel IU takes all pixels of all intersecting pairs into account
    + Matched Pixel IU only takes into account the pixels fro m the matched lines

    TODO: verify that threshold value is not relevant for this metric.

    Args:
        metrics (np.ndarray): metrics matrix, with indices [0..m-1, 0..m-1] for labels 1..m,
            where m is the maximum label in either GT or predicted maps. In channels: intersection
            count, union count, precision, recall.

    Returns:
        tuple: a pair (Pixel IU, Matched Pixel IU)
    """
    label_count_pred, label_count_gt = metrics.shape[:2]

    # find all rows with non-empty intersection
    possible_match_indices = metrics[:,:,0].nonzero()
    match_rows_cols, possible_matches = np.transpose(possible_match_indices), metrics[ possible_match_indices ]
    ious = possible_matches[:,0]/possible_matches[:,1]
    structured_row_col_match_iou = np.array([
        (match_rows_cols[i,0],
         match_rows_cols[i,1],
         possible_matches[i,0],
         possible_matches[i,1],
         possible_matches[i,2],
         possible_matches[i,3],
         ious[i]) for i in range(len(possible_matches))],
         dtype=[('pred_polygon', 'int32'),
                ('gt_polygon', 'int32'),
                ('intersection', 'float32'),
                ('union', 'float32'),
                ('precision', 'float32'),
                ('recall', 'float32'),
                ('iou', 'float32')])

    # pixel-based, page-wide IoU (over all non-empty intersections)
    intersection_count, union_count = [ np.sum(structured_row_col_match_iou[:][field]) for field in ('intersection', 'union') ]
    pixel_iou = intersection_count / union_count

    # pixel-based, page-wide IoU (over all matched pairs)
    matched_intersection_count, matched_union_count = 0, 0

    # sort candidate matches by ascending label order and descending IoU order
    pred_label_iou_copy = structured_row_col_match_iou[['pred_polygon', 'iou']].copy()
    pred_label_iou_copy['iou'] *= -1
    I = np.argsort( pred_label_iou_copy, order=['pred_polygon', 'iou'])

    pred2match = { i:False for i in possible_match_indices[0] }
    for possible_match in structured_row_col_match_iou[I]:
        # ensure that each predicted label is matched to at most one GT label
        if not pred2match[possible_match['pred_polygon']]:
            pred2match[possible_match['pred_polygon']]=True
            matched_intersection_count += possible_match['intersection']
            matched_union_count += possible_match['union']
    matched_pixel_iou = matched_intersection_count / matched_union_count

    return (pixel_iou, matched_pixel_iou)


#def metrics_to_precision_recall_curve( metrics: np.ndarray, threshold_range=np.linspace(0, 1, num=21)) -> np.ndarray:
#    """
#    Compute precision and recalls over a range of IoU thresholds, for plotting purpose.
#
#    Args:
#        metrics (np.ndarray): a 4-channel table with GT labels in rows and predicted labels in columns, where
#                              each entry is a [intersection_count, union_count, precision, recall] sequence.
#        threshold_range: a series of threshold values, between 0 and 1 (default: [0, 0.05, 0.1, ..., 0.95, 1])
#
#    :returns:
#        np.ndarray: a 2D array, with precisions in row 0 and recalls in row 1.
#
#    """
#    precisions_recalls = np.zeros((len(threshold_range), 2))
#    for (i,t) in enumerate(threshold_range):
#        precisions_recalls[i] = metrics_to_aggregate_scores(metrics, iou_threshold=t)[:2]
#        #print(precisions_recalls[:,i])
#    return np.moveaxis( precisions_recalls, 1, 0)


def recover_labels_from_map_value( px: int) -> list:
    """Retrieves intersecting polygon labels from a single map pixel value (for
    diagnosis purpose).

    Args:
        vl (int): a map pixel, whose value is a 32-bit signed integer.

    Returns:
        list: a list of labels
    """
    return [ b for b in np.array( [px], dtype='int32').view('uint8')[::-1] if b ]


def mask_from_polygon_map_functional( polygon_map: Tensor, test: Callable) -> Tensor:
    """Given a 3D map of polygons (where each pixel contains at most 4 labels,
    select labels based on a boolean function.
    Eg. ``mask_from_functional( polygon_map, lambda m: m % 2 )`` covers all odd-labeled
        polygons.

    Args:
        polygon_map (Tensor): polygon set, encoded as a 4-channel, 8-bit tensor.
        test (Callable): a boolean function, to be applied to the map; a partial function 
            may be passed, if added parameters are needed.

    Returns:
        Tensor: a boolean, flat mask.
    """
    if polygon_map.dtype != torch.uint8:
        raise TypeError("First parameter should be a Tensor of uint8.")
    if len(polygon_map.shape) != 3 or polygon_map.shape[0]!=4:
        raise TypeError("Polygon map should have shape (4, m, n)")

    return torch.sum( test( polygon_map ), dim=0).type(torch.bool)


def dummy():
    """Just to check that the module is testable."""
    return True
