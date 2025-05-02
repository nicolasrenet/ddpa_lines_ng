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

from tqdm.auto import tqdm

from PIL import Image
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import random

import sys



def display_annotated_img( img: Tensor, target: dict, alpha=.4, color='g'):
    """ Overlay of instance masks
    Args:
        img (Tensor): (C,H,W) image
        masks (Tensor): (N,H,W) tensor of masks where N=# instances for image
    """
    masks = target['masks']
    boxes = target['boxes']
    bm = torch.sum( masks, axis=0).to(dtype=torch.bool)
    print(bm.shape)
    col = {'r': [1.0,0,0], 'g':[0,1.0,0], 'b':[0,0,1.0]}[color]
    # RED * BOOL * ALPHA
    red_mask = torch.tensor([[col]*img.shape[2]]*img.shape[1]).permute(2,0,1) * bm * alpha
    img_complementary = img * ( ~bm + bm * (1-alpha))
    plt.imshow( (img_complementary + red_mask ).permute(1,2,0))
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
            v2.Resize([1240,1240]),
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

    imgs = list(Path('./dataset').glob('*.jpg'))
    lbls = [ str(img_path).replace('.img.jpg', '.lines.gt.json') for img_path in imgs ]

    # split sets
    imgs_train, imgs_test, lbls_train, lbls_test = split_set( imgs, lbls )
    imgs_train, imgs_val, lbls_train, lbls_val = split_set( imgs_train, lbls_train )
    

    ds_train = ChartersDataset( imgs_train, lbls_train )
    ds_val = ChartersDataset( imgs_val, lbls_val )
    ds_test = ChartersDataset( imgs_test, lbls_test )

    print(ds_train[5][1]['masks'].shape)


    dl_train = DataLoader( ds_train, batch_size=2, collate_fn = lambda b: tuple(zip(*b)))
    dl_val = DataLoader( ds_val, batch_size=2, collate_fn = lambda b: tuple(zip(*b)))

    model = {'net': None, 'train_epochs': [] }

    # single epoch
    model['net'] = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
    model['net'].cuda()
    model['net'].train()
    optimizer =torch.optim.AdamW( model['net'].parameters(), lr=5e-3)
    max_epochs = 10
    
   
    def train_epoch( epoch: int ):
        
        epoch_losses = []

        batches = iter(dl_train)
        for batch_index, sample in enumerate(dl_train):
            imgs, targets = sample
            #print("type(imgs)=", type(imgs))
            imgs = torch.stack(imgs).cuda()
            targets = [ { k:t[k].cuda() for k in ('labels', 'boxes', 'masks') } for t in targets ]
            loss = sum(model['net'](imgs, targets).values())
            
            print('Epoch {}-{}: Training loss: {:.4f}'.format(epoch, batch_index, loss))
            epoch_losses.append( loss.detach() )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        mean_loss = torch.stack( epoch_losses ).mean().item()
        model['train_epochs'].append( {'loss': mean_loss } )

        # validate
        model['net'].eval()
        validation_losses = []
        batches = iter(dl_val)
        for batch_index in tqdm( range(len( batches ))):
            imgs, targets = next(batches)
            imgs = torch.stack(imgs).cuda()
            targets = [ { k:t[k].cuda() for k in ('labels', 'boxes', 'masks') } for t in targets ]
            validation_losses.append( np.sum(model['net'](imgs, targets).values()))
        mean_validation_loss = torch.stack( validation_losses).mean().item()    


        model['net'].train()

        print('Epoch {}: Training loss: {:.4f} - Validation loss: {:.4f}'.format(epoch, mean_loss, mean_validation_loss))
        


    for epoch in range( max_epochs ):

        train_epoch( epoch )

