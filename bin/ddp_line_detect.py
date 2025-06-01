#!/usr/bin/env python3

# nprenet@gmail.com
# 05.2025

"""
Line detection app, that can do either:

    - page-wide line detection, with no consideration for regions
    - region-based line detection, provided a class name (ex. Wr:Oldtext) and the existence of layout segmentation file for each image

The heavy lifting is done by a Mask-RCNN model that computes morphological features for each line map: this script uses them to write a JSON segmentation file.

Output formats: 
    + JSON
    + npy (2D-label map only)

Example call::
    export DIDIP_ROOT=. FSDB_ROOT=~/tmp/data/1000CV
    PYTHONPATH=${DIDIP_ROOT} python3 ./bin/ddp_line_detect -img_paths "${FSDB_ROOT}"/*/*/d9ae9ea49832ed79a2238c2d87cd0765/*seals.crops/*OldText*.jpg -model_path best.mlmodel -mask_classes Wr:OldText

TODO:
    - patch-based detection, where image is processed in two parts.
"""
# stdlib
import sys
from pathlib import Path
import json
import re
import sys
import datetime
import logging

# 3rd party
import torch
from PIL import Image
import skimage as ski
import numpy as np

# Didip
import fargv

# local
src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib


logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)

# tone down unwanted logging
logging.getLogger('PIL').setLevel(logging.INFO)



p = {
        "appname": "lines",
        "model_path": str(src_root.joinpath("best.mlmodel")),
        #"img_paths": set([Path.home().joinpath("tmp/data/1000CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.img.jpg")]),
        "img_paths": set([]),
        "charter_dirs": set(["./"]),
        "mask_classes": [set(['Wr:OldText']), "Names of the seals-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page."],
        "region_segmentation_suffix": [".seals.pred.json", "Regions are given by segmentation file that is <img name stem>.<suffix>."],
        "line_type": [("polygon","legacy_bbox"), "Line segmentation type: polygon = Kraken (CNN-inferred) baselines + polygons; legacy_bbox: legacy Kraken segmentation)"],
        "output_format": [("json", "npy"), "Segmentation output: json=<JSON file>, npy=label map (HW)"],
        'mask_threshold': [.25, "In the post-processing phase, threshold to use for line soft masks."],
}


def build_segdict( img_metadata, segmentation_record, contour_tolerance=4.0 ):
    """
    Construct the line segmentation dictionary (single-region file).

    Args:
        img_metadata (dict): original image's metadata.
        segmentation_record (tuple[np.ndarray, list[tuple]]): a tuple with
            - label map (np.ndarray)
            - a list of line attribute dicts (label, centroid pt, ..., area, polygon_coords)
        contour_tolerance (float): value for contour approximation (default: 4)
    Return:
        dict: a segmentation dictionary
    """
    
    segdict = { 'created': str(datetime.datetime.now()), 'creator': __file__, }
    segdict.update( img_metadata )
    segdict['regions']=[ { 'id': 0, 'type': 'text_region', 'lines': [] } ]

    mp, atts = segmentation_record
    line_id=0
    for att_dict in atts:
        label, polygon_coords, area, line_height, centerline = [ att_dict[k] for k in ('label','polygon_coords','area', 'line_height', 'centerline')]
        #if label==1:
        #    print(polygon_coords[:,1:], "shape=", polygon_coords[:,1:].shape)
        segdict['regions'][0]['lines'].append({ 
                'id': f'l{line_id}', 
                'boundary': ski.measure.approximate_polygon( polygon_coords[:,::-1], tolerance=contour_tolerance).tolist(),
                'stroke_width': int(line_height),
                'centerline': ski.measure.approximate_polygon( centerline[:,::-1], tolerance=contour_tolerance).tolist(),
                })
        line_id += 1
    return segdict


def build_segdict_composite( img_metadata, boxes, segmentation_records, contour_tolerance=4.0):
    """
    Construct the region + line segmentation dictionary.

    Args:
        img_metadata (dict): original image's metadata.
        boxes (list[tuple]): list of LTRB coordinate vectors, one for each region.
        segmentation_records (list[tuple[np.ndarray, list[tuple]]]): a list of N tuples (one
        per region) with
            - label map (np.ndarray)
            - a list of line attribute dicts (label, centroid pt, ..., area, polygon_coords)
        contour_tolerance (float): value for contour approximation (default: 4)

    Return:
        dict: a segmentation dictionary
    """
    
    segdict = { 'created': str(datetime.datetime.now()), 'creator': __file__, }
    segdict.update( img_metadata )
    segdict['regions']=[]

    region_id = 0
    print('build_segdict_composite')
    for box, record in zip(boxes, segmentation_records):
        this_region_lines = []
        line_id = 0
        _, atts = record
        for att_dict in atts:
            label, polygon_coords, area, line_height, centerline = [ att_dict[k] for k in ('label','polygon_coords','area','line_height', 'centerline')]
            this_region_lines.append({
                'id': f'r{region_id}l{line_id}',
                'boundary': ski.measure.approximate_polygon( polygon_coords[:,::-1] + box[:2], tolerance=contour_tolerance).tolist(),
                'stroke_width': int(line_height),
                'centerline': ski.measure.approximate_polygon( centerline[:,::-1] + box[:2], tolerance=contour_tolerance).tolist(),
            })
            line_id += 1
        segdict['regions'].append( { 'id': region_id, 'type': 'text_region', 'boundary': [[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]], 'lines': this_region_lines } )
        region_id += 1
        
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

            # keys from PageXML specs
            img_metadata = {
                    'imageFilename': str(path.name),
                    'imageHeight': img.size[1],
                    'imageWidth': img.size[0],
            }

            output_file_path_wo_suffix = path.parent.joinpath( f'{stem}.{args.appname}.pred' )

            json_file_path = Path(f'{output_file_path_wo_suffix}.json')
            npy_file_path = Path(f'{output_file_path_wo_suffix}.npy')

            import ddp_lineseg as lsg

            if not Path( args.model_path ).exists():
                raise FileNotFoundError("Could not find model file", args.model_path)
            model = lsg.SegModel.load( args.model_path )
            label_map = None

            # Option 1: segment the region crops (from seals), and construct a page-wide file
            if len(args.mask_classes):
                logger.debug(f"Run segmentation on masked regions '{args.mask_classes}', instead of whole page.")
                # parse segmentation file, and extract and concatenate the WritableArea crops
                with open(region_segfile) as regseg_if:
                    regseg = json.load( regseg_if )
                   
                    # iterate over seals crops and segment
                    crops_hwc, boxes, classes = seglib.seals_regseg_to_crops( img, regseg, args.mask_classes )
                    # crop-based predictions: sizes are crops' sizes
                    imgs_t, preds, sizes = lsg.predict( list(crops_hwc), live_model=model )
                    # each segpage: label map, attribute, <image path or id>
                    segmentation_records = [ lsg.post_process( p, orig_size=sz, mask_threshold=args.mask_threshold ) for (p,sz) in zip(preds,sizes) ]
                    label_map = np.squeeze( segmentation_records[0][0] )
                    segdict = build_segdict_composite( img_metadata, boxes, segmentation_records ) 

            # Option 2: single-file segmentation (an Wr:OldText crop, supposedly)
            else:
                logger.info("Starting segmentation")
                imgs_t, preds, sizes = lsg.predict( [img], live_model=model )
                logger.info("Successful segmentation.")
                segmentation_record = lsg.post_process( preds[0], orig_size=sizes[0], mask_threshold=args.mask_threshold )
                label_map = np.squeeze( segmentation_record[0] )
                segdict = build_segdict( img_metadata, segmentation_record )

            ############ 3. Handing the output #################
            output_file_path = Path(f'{output_file_path_wo_suffix}.{args.output_format}')
            logger.debug(f"Serializing segmentation for img.shape={img.size}")

            # JSON file (work from dict)
            with open(output_file_path, 'w') as of:
                if args.output_format == 'json':
                    segdict['image_wh']=img.size
                    json.dump( segdict, of )
                elif args.output_format == 'npy':
                    np.save( output_file_path, label_map )
                logger.info("Segmentation output saved in {}".format( output_file_path ))


