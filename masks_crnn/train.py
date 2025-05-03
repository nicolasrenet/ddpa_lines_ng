#!/usr/bin/env python3

import json
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.tv_tensors import BoundingBoxes, Mask
import torchvision.transforms.v2 as v2

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

from PIL import Image, ImageDraw
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import random

import sys

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


def batch_visuals( imgs:list, results: list[dict], color_count=-1, alpha=.4):

    visuals = []
    imgs = [ img.cpu().numpy() for img in imgs ]
    masks = [ m.detach().cpu().numpy() for m in [ r['masks'] for r in results] ]
    masks = [ m * (m>.4) for m in masks ]
    for img,msk in zip(imgs,masks):
        bm = np.sum( msk, axis=0).astype('bool')
        img_complementary = img * ( ~bm + bm * (1-alpha))
        #print("img_complementary:", img_complementary.shape)
        if color_count==0:
            colors = get_n_color_palette( color_count ) if color_count else get_n_color_palette(len( msk ))
            col_masks = []
            for c,m in zip(colors,msk):
                col_masks.append( np.transpose( np.full( img.shape[1::-1] + (3,), c), (2,0,1)) * m.astype('bool'))
                col_mask = np.sum( np.stack( col_masks), axis=0)
        else:
            #RED * BOOL * ALPHA
            col_canvas = np.full(img.shape[1::-1]+(3,), [0,0,1.0])
            #print("img:", img.shape)
            #print("col_canvas:", col_canvas.shape)
            col_mask = np.transpose( np.full(img.shape[1:]+(3,), [0,0,1.0]),  (2,0,1)) * bm * alpha
        #print("col_mask:", col_mask.shape)
        composed_img_array = np.transpose(img_complementary + col_mask, (1,2,0))
        visuals.append( composed_img_array )
    batched_visuals = np.transpose( np.stack( visuals ), (0,3,1,2))
    print("batched_visuals:", batched_visuals.shape)
    
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
    def __init__(self, img_paths, label_paths, transforms=None):
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
            v2.Resize([680,680]),
            v2.RandomHorizontalFlip(p=1),
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
            polygons = [ [ tuple(p) for p in l['coreBoundary']] for l in segdict['lines'] ]
        
            # Convert polygons to mask images
            masks = Mask(torch.stack([ Mask( ski.draw.polygon2mask( img.size, polyg )).permute(1,0) for polyg in polygons ]))

        # Generate bounding box annotations from segmentation masks
            bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=img.size[::-1])
            #print(bboxes)
            return img, {'masks': masks,'boxes': bboxes, 'labels': labels, 'path': img_path}


if __name__ == '__main__':

    random.seed(46)
    imgs = random.sample( list(Path('./dataset').glob('*.jpg')), 20)
    lbls = [ str(img_path).replace('.img.jpg', '.lines.gt.json') for img_path in imgs ]

    # split sets
    imgs_train, imgs_test, lbls_train, lbls_test = split_set( imgs, lbls )
    imgs_train, imgs_val, lbls_train, lbls_val = split_set( imgs_train, lbls_train )
    

    ds_train = ChartersDataset( imgs_train, lbls_train )
    ds_val = ChartersDataset( imgs_val, lbls_val )
    ds_test = ChartersDataset( imgs_test, lbls_test )

    print(ds_train[5][1]['masks'].shape)


    dl_train = DataLoader( ds_train, batch_size=2, shuffle=True, collate_fn = lambda b: tuple(zip(*b)))
    dl_val = DataLoader( ds_val, batch_size=1, collate_fn = lambda b: tuple(zip(*b)))

    model = {'net': None, 'train_epochs': [] }

    # single epoch
    resume_file = 'last.pt'
    model['net']=maskrcnn_resnet50_fpn_v2(weights=None, num_classes=2)
    if Path(resume_file).exists():
        model['net'].load_state_dict( torch.load(resume_file, weights_only=True))
    
    model['net'].cuda()
    model['net'].train()
    optimizer =torch.optim.AdamW( model['net'].parameters(), lr=5e-3)
    max_epochs = 200
    patience = 50
    best_loss, best_epoch = 100.0, -1

    writer=SummaryWriter()

    def validate(epoch, ):
        validation_losses = []
        batches = iter(dl_val)
        for batch_index in tqdm( range(len( batches ))):
            imgs, targets = next(batches)
            imgs = torch.stack(imgs).cuda()
            targets = [ { k:t[k].cuda() for k in ('labels', 'boxes', 'masks') } for t in targets ]
            loss_dict = model['net'](imgs, targets)
            loss = sum( loss_dict.values()) 
            validation_losses.append( loss.detach())
        mean_validation_loss = torch.stack( validation_losses).mean().item()    
        writer.add_scalar("Loss/val", mean_validation_loss, epoch)
        
        # tensorboard
        model['net'].eval()
        net=model['net'].cpu()
        #inputs = [ ds_val[i][0].cuda() for i in range(2) ]
        inputs = [ ds_val[i][0].cpu() for i in random.sample( range( len(ds_val)), 2) ]
        predictions = net( inputs )
        writer.add_images('batch[10]', batch_visuals( inputs, net( inputs)), 0)
        model['net'].train()

        torch.save(model['net'].state_dict() , 'last.pt')

        return mean_validation_loss
   
    def train_epoch( epoch: int ):
        
        epoch_losses = []
        batches = iter(dl_train)
        for batch_index, sample in enumerate(dl_train):
            imgs, targets = sample
            #print("type(imgs)=", type(imgs))
            imgs = torch.stack(imgs).cuda()
            targets = [ { k:t[k].cuda() for k in ('labels', 'boxes', 'masks') } for t in targets ]
            loss_dict = model['net'](imgs, targets)
            loss = sum( loss_dict.values())
            
            print('Epoch {}-{}: Training loss: {:.4f}'.format(epoch, batch_index, loss))
            epoch_losses.append( loss.detach() )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        mean_loss = torch.stack( epoch_losses ).mean().item()

        writer.add_scalar("Loss/train", mean_loss, epoch)
        model['train_epochs'].append( {'loss': mean_loss } )

        mean_validation_loss = validate( epoch )
        print('Epoch {}: Training loss: {:.4f} - Validation loss: {:.4f}'.format(epoch, mean_loss, mean_validation_loss))



    for epoch in range( max_epochs ):

        validation_loss = train_epoch( epoch )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            torch.save( model['net'].state_dict(), 'best.pt')
        elif epoch - best_epoch > patience:
            print("No improvement since epoch {}: early exit.".format(epoch))
            break

    writer.flush()
    writer.close()
