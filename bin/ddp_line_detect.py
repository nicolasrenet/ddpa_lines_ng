#!/usr/bin/env python3

"""
Read cropped writeable areas produced by the 'seals' app and 
segment it into lines (use a Mask-RCNN engine).

Example call::

    export DIDIP_ROOT=. FSDB_ROOT=~/tmp/data/1000CV
    PYTHONPATH="${HOME}/htr/didipcv/src/:${DIDIP_ROOT}/apps/ddpa_line_detect.py" ${DIDIP_ROOT}/apps/ddpa_lines/bin/ddp_line_detect -img_paths "${FSDB_ROOT}"/*/*/d9ae9ea49832ed79a2238c2d87cd0765/*seals.crops/*OldText*.jpg -model_path best.mlmodel -mask_classes Wr:OldText


Output formats: 

    + JSON

"""
# stdlib
import sys
from pathlib import Path
import json
import re
import sys
import logging

# 3rd party
import torch
from PIL import Image
import skimage as ski

# Didip
import fargv
import json
import numpy as np


src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))

from libs import seglib

#logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


p = {
        "appname": "lines",
        "model_path": str(src_root.joinpath("models/blla.mlmodel")),
        #"img_paths": set([Path.home().joinpath("tmp/data/1000CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.img.jpg")]),
        "img_paths": set([]),
        "charter_dirs": set(["./"]),
        #"mask_classes": [set(['Wr:OldText']), "Names of the seals-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page."],
        "mask_classes": [set([]), "Names of the seals-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page."],
        "region_segmentation_suffix": [".seals.pred.json", "Regions are given by segmentation file that is <img name stem>.<suffix>."],
        "dry_run": False,
        "line_type": [("polygon","legacy_bbox"), "Line segmentation type: polygon = Kraken (CNN-inferred) baselines + polygons; legacy_bbox: legacy Kraken segmentation)"],
        "output_format": [("xml", "json", "pt"), "Segmentation output: xml=<Page XML>, json=<JSON file>, tensor=<a (4,H,W) label map where each pixel can store up to 4 labels (for overlapping polygons)"],
        'mask_threshold': .25,
}


def build_segdict( img, segmentation_record, contour_tolerance=4.0 ):
    """
    TODO: 
        - page-wide keys (name, size, etc.)
        - use area and axis-length to compute line height
        - compute polygon skeleton -> baseline (need to compute straight skeleton, but python implementations do not abound)
    Args:
        img (Image.Image): the original image
        segmentation_record (tuple[np.ndarray, list[tuple]]): a tuple with
            - label map (np.ndarray)
            - a list of line attribute dicts (label, centroid pt, ..., area, polygon_coords)
        contour_tolerance (float): value for contour approximation (default: 4)
    Return:
        dict: a segmentation dictionary
    """

    segdict = {'lines':[] }
    mp, atts = segmentation_record
    for att_dict in atts:
        label, polygon_coords, area, axis_major_length = [ att_dict[k] for k in ('label','coords','area','axis_major_length')]
        #if label==1:
        #    print(polygon_coords[:,1:], "shape=", polygon_coords[:,1:].shape)
        segdict['lines'].append({ 'id': label, 'boundary': ski.measure.approximate_polygon( polygon_coords[:,1:], tolerance=contour_tolerance).tolist()})
    #print(segdict)
    return segdict


def build_segdict_composite( img, boxes, segmentation_records, contour_tolerance=4.0):
    """
    TODO: page-wide keys (name, size, etc.)
        - use area and axis-length to compute line height
        - compute polygon skeleton -> baseline

    Args:
        img (Image.Image): the original image
        boxes (list[tuple]): list of LTRB coordinate vectors, one for each region.
        segmentation_records (list[tuple[np.ndarray, list[tuple]]]): a list of N tuples (one
        per region) with
            - label map (np.ndarray)
            - a list of line attribute dicts (label, centroid pt, ..., area, polygon_coords)
        contour_tolerance (float): value for contour approximation (default: 4)

    Return:
        dict: a segmentation dictionary
    """
    
    segdict = {'lines': [] }
    print('build_segdict_composite')
    for box, record in zip(boxes, segmentation_records):
        _, atts = record
        # adding the box offset
        for att_dict in atts:
            label, polygon_coords, area, axis_major_length = [ att_dict[k] for k in ('label','coords','area','axis_major_length')]
            offset_polygon = ski.measure.approximate_polygon( polygon_coords[:,1:] + box[:2], tolerance=contour_tolerance).tolist()
            segdict['lines'].append( { 'boundary': offset_polygon })
    return segdict
        


if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    all_img_paths = list(sorted(args.img_paths))
    for charter_dir in args.charter_dirs:
        charter_dir_path = Path( charter_dir )
        logger.debug(f"Charter Dir: {charter_dir}")
        if charter_dir_path.is_dir() and charter_dir_path.joinpath("CH.cei.xml").exists():
            charter_images = [str(f) for f in charter_dir_path.glob("*.img.*")]
            all_img_paths += charter_images

        args.img_paths = list(all_img_paths)
        print("AFTER:",args.img_paths)

    logger.debug( args )

    for path in list( args.img_paths ):
        logger.debug( path )
        path = Path(path)

        #stem = Path( path ).stem
        stem = re.sub(r'\..+', '', path.name )

        # only for segmentation on Seals-detected regions
        region_segfile = re.sub(r'.img.jpg', args.region_segmentation_suffix, str(path) )

        with Image.open( path, 'r' ) as img:

            output_file_path_wo_suffix = path.parent.joinpath( f'{stem}.{args.appname}.pred' )

            json_file_path = Path(f'{output_file_path_wo_suffix}.json')
            xml_file_path = Path(f'{output_file_path_wo_suffix}.xml')
            pt_file_path = Path(f'{output_file_path_wo_suffix}.pt')

            import ddp_lineseg as lsg

            if not Path( args.model_path ).exists():
                raise FileNotFoundError("Could not find model file", args.model_path)
            model = lsg.SegModel.load( args.model_path )

            # Option 1: segment the region crops (from seals), and construct a page-wide file
            if len(args.mask_classes):
                logger.debug(f"Run segmentation on masked regions '{args.mask_classes}', instead of whole page.")
                # parse segmentation file, and extract and concatenate the WritableArea crops
                with open(region_segfile) as regseg_if:
                    regseg = json.load( regseg_if )
                   
                    # iterate over seals crops and segment
                    # TODO: ensure seglib.seals_regseg_to_crops() return plain hwc images, as well as crop boundaries (boxes)
                    crops_hwc, boxes, classes = seglib.seals_regseg_to_crops( img, regseg, args.mask_classes )
                    print(crops_hwc, boxes, classes)
                    # crop-based predictions: sizes are crops' sizes
                    imgs_t, preds, sizes = lsg.predict( list(crops_hwc), live_model=model )
                    # each segpage: label map, attribute, <image path or id>
                    segmentation_records = [ lsg.post_process( p, orig_size=sz, mask_threshold=args.mask_threshold ) for (p,sz) in zip(preds,sizes) ]
                    # To generate the segdict, we need:
                    # - the original img
                    # - N crop coordinates
                    # - N crop seg results, each with: label map + dictionary of attributes
                    segdict = build_segdict_composite( img, boxes, segmentation_records ) 

            # Option 2: single-file segmentation (no crops)
            else:
                segmentation_record = None

                logger.info("Starting segmentation")
                imgs_t, preds, sizes = lsg.predict( [img], live_model=model )
                logger.info("Successful segmentation.")
                segmentation_record = lsg.post_process( preds[0], orig_size=sizes[0], mask_threshold=args.mask_threshold )
                segdict = build_segdict( img, segmentation_record )
                #print(segdict)

            ############ 3. Handing the output #################
            output_file_path = Path(f'{output_file_path_wo_suffix}.{args.output_format}')
            logger.debug(f"Serializing segmentation for img.shape={img.size}")

            # JSON file (work from dict)
            with open(output_file_path, 'w') as of:
                segdict['image_wh']=img.size
                json.dump( segdict, of )
                logger.info("Segmentation output saved in {}".format( output_file_path ))






