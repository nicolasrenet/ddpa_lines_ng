#!/usr/bin/env python3
"""


TODO:

+ proper logging
+ time estimate
+ code clean-up

"""

import sys
import json
from pathlib import Path
from typing import Union, Any
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch import nn 
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tvt
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.models.detection import mask_rcnn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import _default_anchorgen, RPNHead, FastRCNNConvFCHead
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tormentor
import math

from tqdm.auto import tqdm

from PIL import Image
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

import fargv

sys.path.append('.')

from libs import segviz, seglib

p = {
    'max_epoch': 250,
    'max_epoch_force': [-1, "If non-negative, this overrides a 'max_epoch' value read from a resumed model"],
    'img_paths': set(list(Path("dataset").glob('*.img.jpg'))),
    'train_set_limit': [0, "If positive, train on a random sampling of the train set."],
    'validation_set_limit': [0, "If positive, validate on a random sampling of the train set (only for 'validate' mode of the script, not for epoch-validation during training)."],
    'line_segmentation_suffix': ".lines.gt.json",
    'polygon_type': 'coreBoundary',
    'backbone': ('resnet101','resnet50'),
    'lr': 2e-4,
    'img_size': [1024, "Resize the input images to <img_size> * <img_size>; if 'img_height non-null, this determines the width."],
    'img_height': [0, "If non-null, input images are resize to <img_size> * <img_height>"],
    'batch_size': 4,
    'patience': 50,
    'tensorboard_sample_size': 2,
    'mode': ('train','validate'),
    'weight_file': None,
    'scheduler': 0,
    'scheduler_patience': 15,
    'scheduler_factor': 0.9,
    'reset_epochs': False,
    'resume_file': 'last.mlmodel',
    'dry_run': False,
}

random.seed(46)


def grid_wave_t( img: Union[np.ndarray,Tensor], grid_cols=(4,20,),random_state=46):
    """
    Makeshift grid-based transform (to be put into a v2.transform). Randomness potentially affects the output
    in 3 ways:
    - range of the sinusoidal function used to move the ys
    - number of columns in the grid
    - amount of the y-offset across xs

    Args:
        img (Union[Tensor,np.ndarray]): input image, as tensor, or numpy array. For the former,
            assume CHW for input, with output the same. For the latter, both input and output 
            are HWC.
        grid_cols (tuple[int]): number of cols for this grid is randomly picked from this tuple.

    Return:
        Union[Tensor,np.ndarray]: tensor or np.ndarray with same size.
    """
    #print("In:", img.shape, type(img), img.dtype)
    # if input is a Tensor, assume CHW and reshape to HWC
    if isinstance(img, Tensor):
        img_t = img.permute(1,2,0) 
        if type(img) is tvt.Image:
            img = tvt.wrap( img_t, like=img)
        else:
            img = img_t
    np.random.seed( random_state ) 
    parallel = np.random.choice([True, False])
    rows, cols = img.shape[0], img.shape[1]

    col_count = np.random.choice(grid_cols)
    #print("grid_wave_t( grid_cols={}, random_state={}, parallel={}, col_count={})".format( grid_cols, random_state, parallel, col_count))
    src_cols = np.linspace(0, cols, col_count)  # simulate folds (increase the number of columns
                                        # for smoother curves
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    offset = float(img.shape[0]/20)
    column_offset = np.random.choice([5,10,20,30], size=src.shape[0]) if (not parallel and col_count <=5) else offset
    dst_rows = src[:, 1] - np.sin(np.linspace(0, np.random.randint(1,13)/4 * np.pi, src.shape[0])) * column_offset
    dst_cols = src[:, 0]
    
    ratio = 1.0 # resulting image is {ratio} bigger that its warped manuscript part 
              # if ratio=1, manuscript is likely to be cropped
    dst_rows = ratio*( dst_rows - offset)
    dst = np.vstack([dst_cols, dst_rows]).T
    tform = ski.transform.PiecewiseAffineTransform()
    tform.estimate(src, dst)
    out_rows = img.shape[0] #- int(ratio * offset)
    out_cols = cols
    out = ski.transform.warp(img, tform, output_shape=(out_rows, out_cols))
    #print("Out of warp():", type(img), "->", type(out), out.dtype, out.shape)
    # keep type, but HWC -> CHW
    if isinstance(img, Tensor):
        out = torch.from_numpy(out).permute(2,0,1)
        if type(img) is tvt.Image:
            out= tvt.wrap( out, like=img)
    #print("Return:", type(out), out.dtype, out.shape)
    return out



class RandomElasticGrid(v2.Transform):
    """
    Deform the image over an elastic grid (v2-compatible)
    """

    def __init__(self, **kwargs):
        """
        Args:
            p (float): prob. for applying the transform.
            grid_cols (tuple[int]): number of columns in the grid from which to pick from (the larger, the smoother the deformation)
                Ex. with (4,20,), the wrapped function randomly picks 4 or 20 columns
        """
        self.params = dict(kwargs)
        self.p = self.params['p']
        # allow to re-seed the wrapped function for each call
        super().__init__()

    def make_params(self, flat_inputs: list[Any]):
        """ Called after initialization """
        apply = (torch.rand(size=(1,)) < self.p).item()
        # s.t. each of the subsequent calls to the wrapped function (on the flattened list of data structures
        # in the sample) uses the _same_ random seeed (but a different one for each batch).
        self.params.update(dict(apply=apply, random_state=random.randint(1,100))) # passed to transform()
        return self.params

    def transform(self, inpt: Any, params: dict['str', Any]):
        if not params['apply']:
            #print('no transform', type(inpt), inpt.dtype)
            return inpt
        if isinstance(inpt, BoundingBoxes):
            return inpt
        return grid_wave_t( inpt, grid_cols=params['grid_cols'], random_state=params['random_state'])




def post_process_two_maps( preds1: dict, preds2: dict, height: int, box_threshold=.9, mask_threshold=.25, orig_size=()):
    """
    NOT FUNCTIONAL Compute lines from predictions.

    Args:
        preds1 (dict[str,torch.Tensor]): predicted dictionary for the top of the page:
            - 'scores'(N) : box probs
            - 'masks' (NHW): line heatmaps
            - 'orig_size': if provided, masks are rescaled to the respective size
        preds2 (dict[str,torch.Tensor]): predicted dictionary for the bottom of the page:
        offset (int): height ratio of patch wr/ original image
    Returns:
        tuple[ np.ndarray, list[tuple[int, list, float, list]]]: a pair with
            - labeled map(1,H,W)
            - a list of line attribute dicts (label, centroid pt, area, polygon coords.)
    """
    # select masks with best box scores
    best_masks_1 = [ m.detach().numpy() for m in preds1['masks'][preds1['scores']>box_threshold]]
    best_masks_2 = [ m.detach().numpy() for m in preds2['masks'][preds2['scores']>box_threshold]]
    # threshold masks
    masks_1 = [ m * (m > mask_threshold) for m in best_masks_1 ]
    masks_2 = [ m * (m > mask_threshold) for m in best_masks_2 ]
    print("masks_1.shape=",masks_1[0].shape)
    print("masks_2.shape=",masks_1[0].shape)

    # merge line masks in each patch
    page_wide_mask_1 = np.sum( masks_1, axis=0 ).astype('bool')
    page_wide_mask_2 = np.sum( masks_1, axis=0 ).astype('bool')
    print("page_wide_mask_1.shape=",page_wide_mask_1.shape)
    print("page_wide mask_2.shape=",page_wide_mask_2.shape)

    # combine 2 overlapping masks
    patch_height, patch_width = page_wide_mask_1.shape[1:]
    page_wide_mask = np.zeros( (1, height, patch_width ))
    page_wide_mask[:,:patch_height]=page_wide_mask_1
    page_wide_mask[:,patch_height:]=page_wide_mask_2

    # optional: scale up masks to the original size of the image
    if orig_size:
        page_wide_mask = ski.transform.resize( page_wide_mask, (1, orig_size[1], orig_size[0]))

    # label components
    labeled_msk = ski.measure.label( page_wide_mask, connectivity=1 )
    
    # sort label from top to bottom (using centroids of labeled regions)
    # note: labels are [1,H,W]. Accordingly, centroids are 3-tuples.
    region_properties = ski.measure.regionprops( labeled_msk )
    # last attribute is an estimate of the line height (Area/<major-axis length>)
    attributes = sorted([ (reg.label, reg.centroid, reg.area, reg.coords, reg.axis_major_length) for reg in region_properties ], key=lambda attributes: (attributes[1][1], attributes[1][2]))
    if [ att[0] for att in attributes ] != list(range(1, np.max(labeled_msk)+1)):
        print("Labels do not follow reading order")
    return (labeled_msk, 
            [ {'label': att[0], 
            'centroid': att[1], 
            'area': att[2], 
            'coords': att[3], 
            'axis_major_length': att[4]} for att in attributes ])

def post_process( preds: dict, box_threshold=.9, mask_threshold=.25, orig_size=()):
    """
    Compute lines from predictions.

    Args:
        preds (dict[str,torch.Tensor]): predicted dictionary for the page:
            - 'scores'(N) : box probs
            - 'masks' (NHW): line heatmaps
            - 'orig_size': if provided, masks are rescaled to the respective size
    Returns:
        tuple[ np.ndarray, list[tuple[int, list, float, list]]]: a pair with
            - labeled map(1,H,W)
            - a list of line attribute dicts (label, centroid pt, area, polygon coords, ...)
    """
    # select masks with best box scores
    best_masks = [ m.detach().numpy() for m in preds['masks'][preds['scores']>box_threshold]]
    # threshold masks
    masks = [ m * (m > mask_threshold) for m in best_masks ]

    # merge masks 
    page_wide_mask_1hw = np.sum( masks, axis=0 ).astype('bool')

    # optional: scale up masks to the original size of the image
    if orig_size:
        page_wide_mask_1hw = ski.transform.resize( page_wide_mask_1hw, (1, orig_size[1], orig_size[0]))

    return get_morphology( page_wide_mask_1hw )

def get_morphology( page_wide_mask_1hw: np.ndarray):
    """
    From a page-wide line mask, extract a labeled map and a dictionary of features.
    
    Args:
        page_wide_mask_1hw (np.ndarray): a binary line mask (1,H,W)
    Returns:
        tuple[ np.ndarray, list[tuple[int, list, float, list]]]: a pair with
            - labeled map(1,H,W)
            - a list of line attribute dicts (label, centroid pt, area, polygon coords, ...)
    """
    # label components
    labeled_msk_1hw = ski.measure.label( page_wide_mask_1hw, connectivity=1 )
    
    # sort label from top to bottom (using centroids of labeled regions) # note: labels are [1,H,W]. Accordingly, centroids are 3-tuples.
    line_region_properties = ski.measure.regionprops( labeled_msk_1hw )
    # list of line attribute tuples 
    attributes = sorted([ (reg.label, reg.centroid, reg.area, reg.coords, reg.axis_major_length) for reg in line_region_properties ], key=lambda attributes: (attributes[1][1], attributes[1][2]))
    
    # compute line heights and center lines from skeletons
    page_wide_skeleton_hw = ski.morphology.skeletonize( page_wide_mask_1hw[0] )
    _, distance = ski.morphology.medial_axis( page_wide_mask_1hw[0], return_distance=True )
    labeled_skl = ski.measure.label( page_wide_skeleton_hw, connectivity=2)
    skeleton_coords = [ reg.coords for reg in ski.measure.regionprops( labeled_skl ) ]
    line_heights = []
    for lbl in range(1, np.max(labeled_skl)+1):
        line_skeleton_dist = page_wide_skeleton_hw * ( labeled_skl == lbl ) * distance 
        line_heights.append( (np.mean(line_skeleton_dist[ line_skeleton_dist != 0])*2).item() )
    assert len(line_heights) == len( line_region_properties ) 

    if [ att[0] for att in attributes ] != list(range(1, np.max(labeled_msk_1hw)+1)):
        print("Labels do not follow reading order")

    return (labeled_msk_1hw, [{
                'label': att[0], 
                'centroid': att[1], 
                'area': att[2], 
                'coords': att[3], 
                #'axis_major_length': att[4],
                'line_height': lh, 
                'centerline': skc,
            } for att,lh,skc in zip(attributes, line_heights, skeleton_coords) ])


def split_set( *arrays, test_size=.2, random_state=46):
    random.seed( random_state)
    seq = range(len(arrays[0]))
    train_set = set(random.sample( seq, int(len(arrays[0])*(1-test_size))))
    test_set = set(seq) - train_set
    sets = []
    for a in arrays:
        sets.extend( [[ a[i] for i in train_set ], [ a[j] for j in test_set ]] )
    return sets

def build_nn( backbone='resnet101'):

    if backbone == 'resnet50':
        return maskrcnn_resnet50_fpn_v2(weights=None, num_classes=2)
        
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=None)#weights=ResNet101_Weights.DEFAULT)
    rpn_anchor_generator = _default_anchorgen()
    #rpn_anchor_generator = AnchorGenerator(sizes=((128,256,512),),
    #                               aspect_ratios=((1.0, 2.0, 4.0, 8.0),))
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    return MaskRCNN( 
            backbone=backbone, num_classes=2,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,)


def predict( imgs: list[Union[str,Path,Image.Image,Tensor,np.ndarray]], live_model=None, model_file='best.mlmodel' ):
    """
    Args:
        imgs (list[Union[Path,Tensor]]): lists of image filenames or tensors; in the latter case, images
            are assumed to have been resized in a previous step (f.i. when predict() is used during the
            training phase and consumes images from the dataset object.
        live_model (SegModel): an instance of the segmentation model class.
        model_file (str): a saved model
    Returns:
        tuple[list[Tensor], list[dict]]: a tuple with 
        - the resized images (as tensors)
        - a list of prediction dictionaries.
        - a list of tuples (width,height) storing size of the original input img
    """
    assert type(imgs) is list

    model = live_model
    if model is None:
        if not Path(model_file).exists():
            return []
        model = SegModel.load( model_file )
    model.net.cpu()
    model.net.eval()

    orig_sizes = []
    img_size = model.hyper_parameters['img_size']
    width, height = (img_size[0],img_size[0]) if len(img_size)==1  else img_size
    tsf = v2.Compose([
        v2.ToImage(),
        v2.Resize([ width,height]),
        v2.ToDtype(torch.float32, scale=True),
    ])
    # every input that is not a tensor needs both resizing and tensor-ification
    if not isinstance(imgs[0], Tensor):
        imgs_live = []
        if isinstance(imgs[0], Path) or type(imgs[0]) is str:
            imgs_live = [ Image.open(img).convert('RGB') for img in imgs ]
        elif isinstance(imgs[0], Image.Image) or type(imgs[0]) is np.ndarray:
            imgs_live = imgs
        imgs, orig_sizes = zip(*[ (tsf(img), img.size) for img in imgs_live ])
    else:
        orig_sizes = [ img.shape[:0:-1] for img in imgs ]
        
    return (imgs, model.net( imgs ), orig_sizes)
    

class SegModel():

    def __init__(self, backbone='resnet101'):
        self.net = build_nn( backbone )
        self.epochs = []
        self.hyper_parameters = {}

    def save( self, file_name ):
        state_dict = self.net.state_dict()
        state_dict['epochs'] = self.epochs
        state_dict['hyper_parameters']=self.hyper_parameters
        torch.save( state_dict, file_name )

    @staticmethod
    def resume(file_name, reset_epochs=False, **kwargs):
        if Path(file_name).exists():
            state_dict = torch.load(file_name, map_location="cpu")
            epochs = state_dict["epochs"]
            del state_dict["epochs"]
            hyper_parameters = state_dict["hyper_parameters"]
            del state_dict['hyper_parameters']

            model = SegModel( hyper_parameters['backbone'] )
            model.net.load_state_dict( state_dict )

            if not reset_epochs:
                model.epochs = epochs if not reset_epochs else []
                model.hyper_parameters = hyper_parameters
            model.net.train()
            model.net.cuda()
            return model
        return SegModel(**kwargs)

    @staticmethod
    def load(file_name, **kwargs):
        if Path(file_name).exists():
            state_dict = torch.load(file_name, map_location="cpu")
            del state_dict["epochs"]
            hyper_parameters = state_dict["hyper_parameters"]
            del state_dict['hyper_parameters']

            model = SegModel( hyper_parameters['backbone'] if 'backbone' in hyper_parameters else 'resnet101')
            model.net.load_state_dict( state_dict )
                                      
            model.hyper_parameters = hyper_parameters
            model.net.eval()
            return model
        return SegModel(**kwargs)






class LineDetectionDataset(Dataset):
    """
    This class represents a PyTorch Dataset for a collection of images and their annotations.
    The class is designed to load images along with their corresponding segmentation masks, bounding box annotations, and labels.
    """
    def __init__(self, img_paths, label_paths, polygon_type='coreBoundary', transforms=None, default_img_size=1024):
        """
        Constructor for the Dataset class.

        Parameters:
            img_paths (list): List of unique identifiers for images.
            label_paths (list): List of label paths.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        super(Dataset, self).__init__()
        
        self._img_paths = img_paths  # List of image keys
        self._label_paths = label_paths  # List of image annotation files
        self.polygon_type = polygon_type
        self._transforms = transforms if transforms is not None else v2.Compose([
            v2.ToImage(),
            v2.Resize([ default_img_size, default_img_size ]),
            v2.ToDtype(torch.float32, scale=True),])

        
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self._img_paths)
        
    def __getitem__(self, index):
        """
        Fetch an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to fetch from the dataset.

        Returns:
            tuple: A tuple containing the image and its associated target (annotations).
        """
        # Retrieve the key for the image at the specified index
        img_path, label_path = self._img_paths[index], self._label_paths[index]
        # Get the annotations for this image
        label_path = Path(str(img_path).replace('.img.jpg','.lines.gt.json'))
        # Load the image and its target (segmentation masks, bounding boxes and labels)
        image, target = self._load_image_and_target(img_path, label_path)
        
        # Apply basic transformations (img -> tensor, resizing, scaling)
        if self._transforms:
            image, target = self._transforms(image, target)

        return image, target

    def _load_image_and_target(self, img_path, annotation_path):
        """
        Load an image and its target (bounding boxes and labels).

        Parameters:
            img_path (Path): image path
            annotation_path (Path): annotation path

        Returns:
            tuple: A tuple containing the image and a dictionary with 'boxes' and 'labels' keys.
        """
        # Open the image file and convert it to RGB
        img = Image.open(img_path)#.convert('RGB')

        with open( annotation_path, 'r') as annotation_if:
            segdict = json.load( annotation_if )
            labels = torch.tensor( [ 1 ]*len(segdict['lines']), dtype=torch.int64)
            polygons = [ [ tuple(p) for p in l[self.polygon_type]] for l in segdict['lines'] ]
            masks = Mask(torch.stack([ Mask( ski.draw.polygon2mask( img.size, polyg )).permute(1,0) for polyg in polygons ]))
            bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=img.size[::-1])
            return img, {'masks': masks, 'boxes': bboxes, 'labels': labels, 'path': img_path, 'orig_size': img.size }

### Training 
if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    hyper_params={ varname:v for varname,v in vars(args).items() if varname in (
        'batch_size', 
        'polygon_type', 
        'backbone',
        'train_set_limit', 
        'lr','scheduler','scheduler_patience','scheduler_factor',
        'max_epoch','patience',)}
    
    hyper_params['img_size']=[ int(args.img_size), int(args.img_size) ] if not args.img_height else [ int(args.img_size), int(args.img_height) ]

    model = SegModel( args.backbone )
    # loading weights only
    if args.weight_file is not None and Path(args.weight_file).exists():
        print('Loading weights from {}'.format( args.weight_file))
        model.net.load_state_dict( torch.load(args.weight_file, weights_only=True))
    # resuming from dictionary
    elif args.resume_file is not None and Path(args.resume_file).exists():
        print('Loading model parameters from resume file {}'.format(args.resume_file))
        model = SegModel.resume( args.resume_file ) # reload hyper-parameters from there
        hyper_params.update( model.hyper_parameters )
    # TODO: partial overriding of param dictionary 
    # elif args.fine_tune
    if args.max_epoch_force >= hyper_params['max_epoch']:
        hyper_params['max_epoch']=args.max_epoch_force

    model.hyper_parameters = hyper_params

    random.seed(46)
    imgs = random.sample( list(args.img_paths), hyper_params['train_set_limit']) if hyper_params['train_set_limit'] else list(args.img_paths)
    lbls = [ str(img_path).replace('.img.jpg', args.line_segmentation_suffix) for img_path in imgs ]

    # split sets
    imgs_train, imgs_test, lbls_train, lbls_test = split_set( imgs, lbls )
    imgs_train, imgs_val, lbls_train, lbls_val = split_set( imgs_train, lbls_train )

#    train_transforms = v2.Compose([
#            v2.ToImage(),
#            v2.Resize( hyper_params['img_size'] ),
#            RandomElasticGrid(p=0.3, grid_cols=(4,20)),
#            v2.RandomRotation( 5 ),
#            v2.RandomHorizontalFlip(p=.2),
#            #v2.SanitizeBoundingBoxes(),
#            v2.ToDtype(torch.float32, scale=True),
#            ])
    basic_transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize( hyper_params['img_size'] ),
            v2.ToDtype(torch.float32, scale=True), ])

    ds_train = LineDetectionDataset( imgs_train, lbls_train, transforms=basic_transforms)
    ds_val = LineDetectionDataset( imgs_val, lbls_val, transforms=basic_transforms)
    ds_test = LineDetectionDataset( imgs_test, lbls_test, transforms=basic_transforms)

    ######################################
    # Tormentor augmentations
    def augment_with_bboxes( sample, aug, device ):
        """ Augment a sample (img + masks), and add bounding boxes to the target

        Args:
            sample (Tuple[Tensor,dict]): tuple with image (as tensor) and label dictionary.
        """
        img, target = sample
        img = img.to(device)
        img = aug(img)
        masks, labels = target['masks'].to(device), target['labels'].to(device)
        # careful: when passing line masks as (L,H,W), Tormentor assumes that L indexes a batch,
        # causing the transform to be called with different parameters for each line mask. Solution:
        # augment each mask separately 
        #masks = aug(target['masks']).to(device)
        masks = torch.stack( [ aug(m, is_mask=True) for m in target['masks'] ], axis=0).to(device)
        # first, filter empty masks
        keep = torch.sum( masks, dim=(1,2)) > 10
        masks, labels = masks[keep], labels[keep]
        # construct boxes, filter out invalid ones
        boxes=BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=img.shape)
        keep=(boxes[:,0]-boxes[:,2])*(boxes[:,1]-boxes[:,3]) != 0

        target['boxes'], target['labels'], target['masks'] = boxes[keep], labels[keep], masks[keep]
        return (img, target)

    augRotate = tormentor.Rotate.override_distributions(radians=tormentor.Uniform((-math.pi/8, math.pi/8)))
    # first augmentation in the list is a pass-through
    aug = tormentor.AugmentationChoice.create( [ tormentor.StaticImageAugmentation, tormentor.Wrap, augRotate, tormentor.Perspective ] )
    ds_train = tormentor.AugmentedDs( ds_train, tormentor.aug, computation_device='cuda', augment_sample_function=augment_with_bboxes )
    #################################

    dl_train = DataLoader( ds_train, batch_size=hyper_params['batch_size'], shuffle=True, collate_fn = lambda b: tuple(zip(*b)))
    dl_val = DataLoader( ds_val, batch_size=1, collate_fn = lambda b: tuple(zip(*b)))

    # update learning parameters from past epochs
    best_loss, best_epoch, lr = np.inf, -1, hyper_params['lr']
    if model.epochs:
        best_epoch,  best_loss = min([ (i, ep['validation_loss']) for i,ep in enumerate(model.epochs) ], key=lambda t: t[1])
        if 'lr' in model.epochs[-1]:
            lr = model.epochs[-1]['lr']
            print("Read start lR from last stored epoch: {}".format(lr))
    print(f"Best validation loss ({best_loss}) at epoch {best_epoch}")

    optimizer = torch.optim.AdamW( model.net.parameters(), lr=lr )
    scheduler = ReduceLROnPlateau( optimizer, patience=hyper_params['scheduler_patience'], factor=hyper_params['scheduler_factor'] )

    def validate():
        validation_losses = []
        loss_box_reg = []
        loss_mask = []
        batches = iter(dl_val)
        for batch_index in (pbar := tqdm( range(len( batches )))):
            pbar.set_description('Validate')
            imgs, targets = next(batches)
            imgs = torch.stack(imgs).cuda()
            targets = [ { k:t[k].cuda() for k in ('labels', 'boxes', 'masks') } for t in targets ]
            loss_dict = model.net(imgs, targets)
            loss = sum( loss_dict.values()) 
            validation_losses.append( loss.detach())
            loss_box_reg.append( loss_dict['loss_box_reg'].detach())
            loss_mask.append( loss_dict['loss_mask'].detach())
        print( "Loss boxes: {}".format( torch.stack(loss_box_reg).mean().item()))
        print( "Loss masks: {}".format( torch.stack(loss_mask).mean().item()))
        return torch.stack( validation_losses ).mean().item()    
        
    def update_tensorboard(writer, epoch, training_loss, validation_loss):
        writer.add_scalar("Loss/train", training_loss, epoch)
        writer.add_scalar("Loss/val", validation_loss, epoch)
        model.net.eval()
        net=model.net.cpu()
        inputs = [ ds_val[i][0].cpu() for i in random.sample( range( len(ds_val)), args.tensorboard_sample_size) ]
        predictions = net( inputs )
        # (H,W,C) -> (C,H,W)
        #writer.add_images('batch[10]', np.transpose( batch_visuals( inputs, net( inputs ), color_count=5), (0,3,1,2)))
        model.net.cuda()
        model.net.train()
   
    def train_epoch( epoch: int ):
        
        epoch_losses = []
        batches = iter(dl_train)
        
        for batch_index, batch in enumerate(pbar := tqdm(dl_train)):
            pbar.set_description(f'Epoch {epoch}')
            imgs, targets = batch
            imgs = torch.stack(imgs).cuda()
            targets = [ { k:t[k].cuda() for k in ('labels', 'boxes', 'masks') } for t in targets ]
            loss_dict = model.net(imgs, targets)
            loss = sum( loss_dict.values())
            
            epoch_losses.append( loss.detach() )
            loss.backward()
                
            # display gradient
            # plt.imshow( imgs[0].grad.permute(1,2,0) )

            optimizer.step()
            optimizer.zero_grad()

        return torch.stack( epoch_losses ).mean().item()

    if args.mode == 'train':
        
        model.net.cuda()
        model.net.train()
            
        writer=SummaryWriter()

        epoch_start = len( model.epochs )
        if epoch_start > 0:
            print(f"Resuming training at epoch {epoch_start}.")

        for epoch in range( epoch_start, 0 if args.dry_run else hyper_params['max_epoch'] ):

            mean_training_loss = train_epoch( epoch )
            mean_validation_loss = validate()

            update_tensorboard(writer, epoch, mean_training_loss, mean_validation_loss)

            if hyper_params['scheduler']:
                scheduler.step( mean_validation_loss )
            model.epochs.append( {
                'training_loss': mean_training_loss, 
                'validation_loss': mean_validation_loss,
                'lr': scheduler.get_last_lr()[0]
            } )
            torch.save(model.net.state_dict() , 'last.pt')
            model.save('last.mlmodel')

            if mean_validation_loss < best_loss:
                print("Mean validation loss ({}) < best loss ({}): updating best model.".format(mean_validation_loss, best_loss))
                best_loss = mean_validation_loss
                best_epoch = epoch
                torch.save( model.net.state_dict(), 'best.pt')
                model.save( 'best.mlmodel' )
            print('Training loss: {:.4f} (lr={}) - Validation loss: {:.4f} - Best epoch: {} (loss={:.4f})'.format(
                mean_training_loss, 
                scheduler.get_last_lr()[0],
                mean_validation_loss, 
                best_epoch,
                best_loss,))
            if epoch - best_epoch > hyper_params['patience']:
                print("No improvement since epoch {}: early exit.".format(best_epoch))
                break

        writer.flush()
        writer.close()

    # validation + metrics
    elif args.mode == 'validate':
        # 1st pass: mask-rcnn validation, for loss
        mean_validation_loss = validate()
        print('Validation loss: {:.4f}'.format(mean_validation_loss))

        # 2nd pass: metrics on post-processed lines
        # use the same model
        pms = []
        for i,sample in enumerate(list(ds_val)[:args.validation_set_limit] if args.validation_set_limit else ds_val):
            img, target = sample
            print(f'{i}: computing gt_map...', end='')
            gt_map = segviz.gt_masks_to_labeled_map( target['masks'] )
            print(f'computing pred_map...', end='')
            imgs, preds, _ = predict( [img], live_model=model ) 
            pred_map = np.squeeze(post_process( preds[0], mask_threshold=.2, box_threshold=.75 )[0]) 
            print(f'computing pixel_metrics')
            pms.append( seglib.polygon_pixel_metrics_two_flat_maps( pred_map, gt_map ))
        print(seglib.mAP( pms ))

        tps, fps, fns = zip(*[ seglib.polygon_pixel_metrics_to_line_based_scores_icdar_2017( pm )[:3] for pm in pms ])
        print("ICDAR 2017")
        print("F1: {}".format( 2.0 * (sum(tps) / (2*sum(tps)+sum(fps)+sum(fns)))))
        print("Jaccard: {}".format(  (sum(tps) / (sum(tps)+sum(fps)+sum(fns)))))

            





    
