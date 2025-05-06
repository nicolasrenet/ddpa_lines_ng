#!/usr/bin/env python3
"""


TODO:

+ time estimate
+ visuals: heatmaps, boxes


"""

import sys
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch import nn 

import torchvision
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.models.detection import mask_rcnn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import _default_anchorgen, RPNHead, FastRCNNConvFCHead
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
import fargv

p = {
    'max_epoch': 250,
    'img_paths': list(Path("dataset").glob('*.img.jpg')),
    'train_set_limit': 0,
    'line_segmentation_suffix': ".lines.gt.json",
    'polygon_type': 'coreBoundary',
    'lr': 2e-4,
    'img_size': 1240,
    'batch_size': 8,
    'patience': 50,
    'tensorboard_sample_size': 2,
    'mode': ('train','validate'),
    'weight_file': None,
    'scheduler': 0,
    'scheduler_patience': 20,
    'scheduler_factor': 0.5,
    'reset_epochs': False,
    'resume_file': 'last.mlmodel',
    'dry_run': False,
}

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


def batch_visuals( imgs:list, results: list[dict], threshold=.5, color_count=-1, alpha=.4):

    visuals = []
    imgs = [ img.cpu().numpy() for img in imgs ]
    masks = [ m.detach().cpu().numpy() for m in [ r['masks'] for r in results] ]
    masks = [ m * (m>threshold) for m in masks ]
    for img,msk in zip(imgs,masks):
        # (C,H,W) -> (H,W,C)
        bm = np.transpose( np.sum( msk, axis=0).astype('bool'), (1,2,0))
        img = np.transpose( img, (1,2,0))
        img_complementary = img * ( ~bm + bm * (1-alpha))
        col_msk = None
        if color_count>=0:
            labeled_msk = ski.measure.label( bm, connectivity=1)
            colors = get_n_color_palette( color_count ) if color_count > 0 else get_n_color_palette( np.max(labeled_msk))
            col_msk = np.zeros( img.shape, dtype=img.dtype )
            for l in range(1, np.max(labeled_msk)+1):
                col = np.array(colors[l % len(colors) ])
                col_msk += (labeled_msk==l) * (col/255.0)
            col_msk *= alpha
        else:
            # RED * BOOL * ALPHA
            col_msk = np.full(img.shape, [0,0,1.0]) * bm * alpha
        composed_img_array = img_complementary + col_msk
        # Combination: (H,W,C), i.e. fit for image viewers and plots
        visuals.append( composed_img_array )
    batched_visuals = np.stack( visuals )#, (0,3,1,2))
    
    return batched_visuals

def display_annotated_img( img: Tensor, target: dict, alpha=.4, color='g'):
    """ Overlay of instance masks
    Args:
        img (Tensor): (C,H,W) image
        masks (Tensor): (N,H,W) tensor of masks where N=# instances for image
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


def split_set( *arrays, test_size=.2, random_state =46):
    random.seed( random_state)
    seq = range(len(arrays[0]))
    train_set = set(random.sample( seq, int(len(arrays[0])*(1-test_size))))
    test_set = set(seq) - train_set
    sets = []
    for a in arrays:
        sets.extend( [[ a[i] for i in train_set ], [ a[j] for j in test_set ]] )
    return sets


class ChartersDataset(Dataset):
    """
    This class represents a PyTorch Dataset for a collection of images and their annotations.
    The class is designed to load images along with their corresponding segmentation masks, bounding box annotations, and labels.
    """
    def __init__(self, img_paths, label_paths, img_size, transforms=None):
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
        self._transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize([ img_size, img_size]),
            v2.RandomHorizontalFlip(p=.2),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
            ])

        
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

        Parameters:
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
        
        # Apply the transformations, if any
        if self._transforms:
            image, target = self._transforms(image, target)
        
        target['labels'].to(dtype=torch.int64)

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
        img = Image.open(img_path).convert('RGB')

        with open( annotation_path, 'r') as annotation_if:
            segdict = json.load( annotation_if )
            labels = torch.tensor( [ 1 ]*len(segdict['lines']), dtype=torch.int64)
            #print(type(labels), labels.dtype)
            polygons = [ [ tuple(p) for p in l[hyper_params['polygon_type']]] for l in segdict['lines'] ]
            # Convert polygons to mask images
            masks = Mask(torch.stack([ Mask( ski.draw.polygon2mask( img.size, polyg )).permute(1,0) for polyg in polygons ]))
            # Generate bounding box annotations from segmentation masks
            bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=img.size[::-1])
            return img, {'masks': masks,'boxes': bboxes, 'labels': labels, 'path': img_path}

def build_nn( bb='resnet101'):

    if bb=='resnet50':
        return maskrcnn_resnet50_fpn_v2(weights=None, num_classes=2)
        
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=None)#weights=ResNet101_Weights.DEFAULT)
    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    return MaskRCNN( 
            backbone=backbone, num_classes=2,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,)


def predict( imgs: Path, model_file='best.mlmodel' ):

    if not Path(model_file).exists():
        return []

    model = SegModel.load( model_file )
    img_size = model.hyper_parameters['img_size']
    
    tsf = v2.Compose([
        v2.ToImage(),
        v2.Resize([ img_size, img_size]),
        v2.ToDtype(torch.float32, scale=True),
    ])
    imgs = [ tsf( Image.open(img).convert('RGB')) for img in imgs ]
    return (imgs, model.net( imgs ))
    

class SegModel():

    def __init__(self, hyper_params={}):
        self.net = build_nn()
        self.epochs = []
        self.hyper_parameters = hyper_params

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

            model = SegModel()
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

            model = SegModel()
            model.net.load_state_dict( state_dict )
                                      
            model.hyper_parameters = hyper_parameters
            model.net.eval()
            return model
        return SegModel(**kwargs)

### Training 
if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    hyper_params={ varname:v for varname,v in vars(args).items() if varname in (
        'img_size', 
        'batch_size', 
        'polygon_type', 
        'train_set_limit', 
        'lr','scheduler','scheduler_patience','scheduler_factor',
        'max_epoch','patience',)}

    model = SegModel()

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

    model.hyper_parameters = hyper_params
            
    random.seed(46)
    imgs = random.sample( args.img_paths, hyper_params['train_set_limit']) if hyper_params['train_set_limit'] else args.img_paths
    lbls = [ str(img_path).replace('.img.jpg', args.line_segmentation_suffix) for img_path in imgs ]

    # split sets
    imgs_train, imgs_test, lbls_train, lbls_test = split_set( imgs, lbls )
    imgs_train, imgs_val, lbls_train, lbls_val = split_set( imgs_train, lbls_train )

    ds_train = ChartersDataset( imgs_train, lbls_train, hyper_params['img_size'] )
    ds_val = ChartersDataset( imgs_val, lbls_val, hyper_params['img_size'] )
    ds_test = ChartersDataset( imgs_test, lbls_test, hyper_params['img_size'] )

    dl_train = DataLoader( ds_train, batch_size=hyper_params['batch_size'], shuffle=True, collate_fn = lambda b: tuple(zip(*b)))
    dl_val = DataLoader( ds_val, batch_size=1, collate_fn = lambda b: tuple(zip(*b)))

    optimizer = torch.optim.AdamW( model.net.parameters(), lr=hyper_params['lr'])
    scheduler = ReduceLROnPlateau( optimizer, patience=hyper_params['scheduler_patience'], factor=hyper_params['scheduler_factor'] )
    best_loss, best_epoch = np.inf, -1
    if model.epochs:
        best_loss,  best_epoch = min([ (i, ep['validation_loss']) for i,ep in enumerate(model.epochs) ], key=lambda t: t[1])
    print(best_loss, best_epoch)


    def validate():
        validation_losses = []
        batches = iter(dl_val)
        for batch_index in (pbar := tqdm( range(len( batches )))):
            pbar.set_description('Validate')
            imgs, targets = next(batches)
            imgs = torch.stack(imgs).cuda()
            targets = [ { k:t[k].cuda() for k in ('labels', 'boxes', 'masks') } for t in targets ]
            loss_dict = model.net(imgs, targets)
            loss = sum( loss_dict.values()) 
            validation_losses.append( loss.detach())
        return torch.stack( validation_losses).mean().item()    
        
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
        
        for batch_index, sample in enumerate(pbar := tqdm(dl_train)):
            pbar.set_description(f'Epoch {epoch}')
            imgs, targets = sample
            imgs = torch.stack(imgs).cuda()
            targets = [ { k:t[k].cuda() for k in ('labels', 'boxes', 'masks') } for t in targets ]
            loss_dict = model.net(imgs, targets)
            loss = sum( loss_dict.values())
            
            epoch_losses.append( loss.detach() )
            loss.backward()
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
                'validation_loss': mean_validation_loss 
            } )
            torch.save(model.net.state_dict() , 'last.pt')
            model.save('last.mlmodel')

            if mean_validation_loss < best_loss:
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

    elif args.mode == 'validate':
        mean_validation_loss = validate(-1)
        print('Validation loss: {:.4f}'.format(mean_validation_loss))

