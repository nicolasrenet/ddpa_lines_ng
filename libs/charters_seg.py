
# stdlib
import sys
import warnings
import random
import tarfile
import json
import shutil
import re
import os
from pathlib import *
from typing import *

# 3rd-party
from tqdm import tqdm
import defusedxml.ElementTree as ET
import skimage as ski
#import xml.etree.ElementTree as ET
from PIL import Image, ImagePath
import gzip

import numpy as np
import torch
from torch import Tensor
import torchvision
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

from . import download_utils as du
from . import seglib


class DataException( Exception ):
    """"""
    pass


"""
Utility classes to manage charter data.

"""

import logging
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)



class ChartersDataset(VisionDataset):
    """A generic dataset class for charters, equipped with a rich set of methods for segmentation tasks:

        * region extraction methods (from original page images and XML metadata)

        Attributes:
            dataset_resource (dict): meta-data (URL, archive name, type of repository).

            work_folder_name (str): The work folder is where a task-specific instance of the data is created; if it not
                passed to the constructor, a default path is constructed, using this default name.

            root_folder_basename (str): A basename for the root folder, that contains
                * the archive
                * the subfolder that is created from it.
                * work folders for specific tasks
                By default, a folder named ``data/<root_folder_basename>`` is created in the *project directory*,
                if no other path is passed to the constructor.

    """

    dataset_resource = None

    work_folder_name = "ChartersHandwritingDataset"

    root_folder_basename="Charters"

    def __init__( self,
                root: str='',
                work_folder: str = '', # here further files are created, for any particular task
                subset: str = 'train',
                subset_ratios: Tuple[float,float,float]=(.7, 0.1, 0.2),
                transform: Optional[Callable] = None,
                extract_pages: bool = False,
                from_page_tsv_file: str = '',
                from_page_dir: str = '',
                from_work_folder: str = '',
                build_items: bool = True,
                count: int = 0,
                resume_task: bool = False,
                gt_suffix:str = 'lines.gt.json',
                polygon_key:str = 'boundary',
                ) -> None:
        """Initialize a dataset instance.

        Args:
            root (str): Where the archive is to be downloaded and the subfolder containing
                original files (pageXML documents and page images) is to be created. 
                Default: subfolder `data/Charters' in this project's directory.
            work_folder (str): Where line images and ground truth transcriptions fitting a
                particular task are to be created; default: '<root>/ChartersHandwritingDatasetSegment';
                if parameter is a relative path, the work folder is created under
                <root>; an absolute path overrides this.
            subset (str): 'train' (default), 'validate' or 'test'.
            subset_ratios (Tuple[float, float, float]): ratios for respective ('train', 
                'validate', ...) subsets
            transform (Callable): Function to apply to the PIL image at loading time.
            extract_pages (bool): if True, extract the archive's content into the base
                folder no matter what; otherwise (default), check first for a file tree 
                with matching name and checksum.
            shape (str): Extract each line as bbox ('bbox': default), 
                bbox+mask (2 files, 'mask'), or as padded polygons ('polygon')
            build_items (bool): if True (default), extract and store images for the task
                from the pages; otherwise, just extract the original data from the archive.
            from_page_tsv_file (str): if set, the data are to be loaded from the given file
                (containing folder is assumed to be the work folder, superceding the
                work_folder option).
            from_page_dir (str): if set, the samples have to be extracted from the 
                raw page data contained in the given directory.
            from_work_folder (str): if set, the samples are to be loaded from the 
                given directory, without prior processing.
            count (int): Stops after extracting {count} image items (for testing 
                purpose only).
            resume_task (bool): If True, the work folder is not purged. Only those page
                items (lines, regions) that not already in the work folder are extracted.
                (Partially implemented: works only for lines.)
            gt_suffix (str): 'xml' for PageXML (default) or valid, unique suffix of JSON file.
                Ex. 'htr.gt.json'
            polygon_key (str): in the input segmentation dictionary, key for the polygon boundaries:
                'boundary' (default)

        """

        # A dataset resource dictionary needed, unless we build from existing files
        if self.dataset_resource is None and not (from_page_dir or from_page_tsv_file or from_work_folder):
            raise FileNotFoundError("In order to create a dataset instance, you need either:" +
                                    "\n\t + a valid resource dictionary (cf. 'dataset_resource' class attribute)" +
                                    "\n\t + one of the following options: -from_page_dir, -from_work_folder, -from_page_tsv_file")
        
        trf = v2.Compose( [ v2.ToDtype(torch.float32, scale=True) ])
        if transform is not None:
            trf = v2.Compose( [ trf, transform ] )
        super().__init__(root, transform=trf ) 

        self.root = Path(root) if root else Path(__file__).parents[1].joinpath('data', self.root_folder_basename)

        logger.debug("Root folder: {}".format( self.root ))
        if not self.root.exists():
            self.root.mkdir( parents=True )
            logger.debug("Create root path: {}".format(self.root))

        self.raw_data_folder_path = None
        self.work_folder_path = None 

        self.from_page_tsv_file = ''
        if from_page_tsv_file == '':
            # Local file system with data samples, no archive
            if from_work_folder != '':
                work_folder = from_work_folder
                logger.debug("work_folder="+ work_folder)
                if not Path(work_folder).exists():
                    raise FileNotFoundError(f"Work folder {self.work_folder_path} does not exist. Abort.")
                
            # Local file system with raw page data, no archive 
            elif from_page_dir != '':
                self.raw_data_folder_path = Path( from_page_dir )
                if not self.raw_data_folder_path.exists():
                    raise FileNotFoundError(f"Directory {self.raw_data_folder_path} does not exist. Abort.")
                self.pages = sorted( self.raw_data_folder_path.glob('*.{}'.format(gt_suffix)))

            # Online archive
            elif self.dataset_resource is not None:
                # tarball creates its own base folder
                self.raw_data_folder_path = self.root.joinpath( self.dataset_resource['tarball_root_name'] )
                self.download_and_extract( self.root, self.root, self.dataset_resource, extract_pages )
                # input PageXML files are at the root of the resulting tree
                #        (sorting is necessary for deterministic output)
                self.pages = sorted( self.raw_data_folder_path.glob('*.{}'.format(gt_suffix)))
            else:
                raise FileNotFoundError("Could not find a dataset source!")
        else:
            # used only by __str__ method
            self.from_page_tsv_file = from_page_tsv_file

        self.config = {
                'count': count,
                'resume_task': resume_task,
                'from_page_tsv_file': from_page_tsv_file,
                'subset': subset,
                'subset_ratios': subset_ratios,
                'gt_suffix': gt_suffix,
                'polygon_key': polygon_key,
        }

        self.data = []

        if (from_page_tsv_file != '' or from_work_folder!=''):
            build_items = False
        logger.info(f"build_items={build_items}")

        if from_work_folder and not from_page_tsv_file:
            for ss in ('train', 'validate', 'test'):
                if ss==subset:
                    continue
                data = self._build_task(build_items=False, work_folder=work_folder, subset=ss )
                self.dump_data_to_tsv(data, Path(self.work_folder_path.joinpath(f"charters_ds_{ss}.tsv")) )
        
        self.data = self._build_task(build_items=build_items, work_folder=work_folder, subset=subset )
        if self.data and not from_page_tsv_file:
            # Generate a TSV file with one entry per img/transcription pair
            self.dump_data_to_tsv(self.data, Path(self.work_folder_path.joinpath(f"charters_ds_{subset}.tsv")) )
        
        self._generate_readme("README.md", 
                    { 'subset': subset,
                      'subset_ratios': subset_ratios, 
                      'build_items': build_items, 
                      'count': count, 
                      'from_page_tsv_file': from_page_tsv_file,
                      'from_page_dir': from_page_dir,
                      'from_work_folder': from_work_folder,
                      'work_folder': work_folder, 
                     } )

    def download_and_extract(
            self,
            root: Path,
            raw_data_folder_path: Path,
            fl_meta: dict,
            extract=False) -> None:
        """Download the archive and extract it. If a valid archive already exists in the root location,
        extract only.

        Args:
            root (Path): where to save the archive raw_data_folder_path (Path): where to extract the archive.
            fl_meta (dict): a dictionary with file meta-info (keys: url, filename, md5, full-md5, origin, desc)
            extract (bool): If False (default), skip archive extraction step.

        Returns:
            None

        Raises:
            OSError: the base folder does not exist.
        """
        output_file_path = None
        print(fl_meta)
        # downloadable archive
        if 'url' in fl_meta:
            output_file_path = root.joinpath( fl_meta['tarball_filename'])

            if 'md5' not in fl_meta or not du.is_valid_archive(output_file_path, fl_meta['md5']):
                logger.info("Downloading archive...")
                du.resumable_download(fl_meta['url'], root, fl_meta['tarball_filename'], google=(fl_meta['origin']=='google'))
            else:
                logger.info("Found valid archive {} (MD5: {})".format( output_file_path, self.dataset_resource['md5']))
        elif 'file' in fl_meta:
            output_file_path = Path(fl_meta['file'])

        if not raw_data_folder_path.exists() or not raw_data_folder_path.is_dir():
            raise OSError("Base folder does not exist! Aborting.")

        # skip if archive already extracted (unless explicit override)
        if not extract: # and du.check_extracted( raw_data_folder_path.joinpath( self.dataset_resource['tarball_root_name'] ) , fl_meta['full-md5'] ):
            logger.info('Found valid file tree in {}: skipping the extraction stage.'.format(str(raw_data_folder_path.joinpath( self.dataset_resource['tarball_root_name'] ))))
            return
        if output_file_path.suffix == '.tgz' or output_file_path.suffixes == [ '.tar', '.gz' ] :
            with tarfile.open(output_file_path, 'r:gz') as archive:
                logger.info('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( raw_data_folder_path )
        # task description
        elif output_file_path.suffix == '.zip':
            with zipfile.ZipFile(output_file_path, 'r' ) as archive:
                logger.info('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( raw_data_folder_path )


    def _build_task( self, 
                   build_items: bool=True, 
                   work_folder: str='', 
                   subset: str='train', 
                   )->List[dict]:
        """Build the image/GT samples required for an HTR task, either from the raw files (extracted from archive)
        or a work folder that already contains compiled files.

        Args:
            build_items (bool): if True (default), go through the compilation step; otherwise, work from the existing work folder's content.
            work_folder (str): Where line images and ground truth transcriptions fitting a particular task
                are to be created; default: './MonasteriumHandwritingDatasetHTR'.
            subset (str): sample subset to be returned - 'train' (default), 'validate' or 'test'; 

        Returns:
            List[dict]: a list of dictionaries.

        Raises:
            FileNotFoundError: the TSV file passed to the `from_line_tsv_file` option does not exist.
        """
        if self.config['from_page_tsv_file'] != '':
            tsv_path = Path( self.config['from_page_tsv_file'] )
            if tsv_path.exists():
                self.work_folder_path = tsv_path.parent
                # paths are assumed to be absolute
                self.data = self.load_from_tsv( tsv_path )
                logger.debug("data={}".format( self.data[:6]))
            else:
                raise FileNotFoundError(f'File {tsv_path} does not exist!')
        else:
            if work_folder=='':
                self.work_folder_path = Path('data', self.work_folder_name+'Segment') 
                logger.debug("Setting default location for work folder: {}".format( self.work_folder_path ))
            else:
                # if work folder is an absolute path, it overrides the root
                self.work_folder_path = Path( work_folder )
                logger.debug("Work folder: {}".format( self.work_folder_path ))

            if not self.work_folder_path.is_dir():
                self.work_folder_path.mkdir(parents=True)
                logger.debug("Creating work folder = {}".format( self.work_folder_path ))

        if build_items:
                samples = self._build_line_seg_samples( self.raw_data_folder_path, self.work_folder_path)
        else:
            logger.info("Building samples from existing images and masks in {}".format(self.work_folder_path))
            samples = self.load_items_from_dir( self.work_folder_path )

        data = self._split_set( samples, ratios=self.config['subset_ratios'], subset=subset)
        logger.info(f"Subset '{subset}' contains {len(data)} samples.")

        return data

    @staticmethod
    def load_items_from_dir( work_folder_path: Union[Path,str] ) -> List[dict]:
        """ 
        Construct a list of samples from a directory that has been populated with page images
        and tensors of metadata (boxes and masks). 

        Args:
            work_folder_path (Union[Path,str]): a folder containing images (`*.png`, `*.img.jpg`, or
                `*.jpg`), tensors of masks (`*.masks.npy.gz`), and tensors of bboxes coordinates
                (`*.boxes.npy.gz`).
        Returns:
            List[Tuple[Path,Dict[str,np.ndarray]]]: a list of sample pairs, with the input image
            path as the first element and a dictionary of metadata tensors as the second element.
        """
        samples = []
        for mask_path in work_folder_path.glob('*.masks.npy.gz'):
            sample_dict = {}
            page_prefix = re.sub(r'\..+', '', str(mask_path))
            with gzip.GzipFile( mask_path, 'r') as zf:
                sample_dict['masks']=np.load( zf )
            with gzip.GzipFile( Path(page_prefix).with_suffix('.boxes.npy.gz'), 'r') as zf:
                sample_dict['boxes']=np.load( zf )

            img_suffix = ''
            for img_suffix in ('.img.jpg', '.jpg', '.png'):
                if Path(page_prefix).with_suffix( img_suffix ).exists():
                    break
            if not img_suffix:
                raise FileNotFoundError("Could not find an image file.")
            samples.append( ( Path(page_prefix).with_suffix(suffix), sample_dict))
        return samples


    @staticmethod
    def load_items_from_tsv( file_path: Union[Path,str] ) -> List[dict]:
        """ Load samples from an existing TSV file. Each input is a tuple::

            <page img filename> <name of the tensor of box coordinates> <name of the tensor of masks>

        Args:
            file_path (Path): A file path.
        Returns:
            List[Tuple[Path,Dict[str,np.ndarray]]]: a list of sample pairs, with the input image
                path as the first element and a dictionary of metadata tensors as the second element.
        """
        samples = []
        with open( file_path, 'r') as infile:
            for line in infile:
                sample_dict = {}
                img_filename, box_filename, mask_filename = line[:-1].split('\t')
                with gzip.GzipFile( mask_filename, 'r') as zf:
                    sample_dict['masks']=np.load( zf )
                with gzip.GzipFile( box_filename, 'r') as zf:
                    sample_dict['boxes']=np.load( zf )
                samples.append( (Path(img_filename), sample_dict ))
        return samples


    @staticmethod
    def dataset_stats( samples: List[dict] ) -> str:
        """Compute basic stats about sample sets.

        + avg, median, min, max on image heights and widths
        + avg, median, min, max on number of boxes

        Args:
            samples (List[dict]): a list of samples.

        Returns:
            str: a string.
        """
        heights = [ np.mean(s[1]["boxes"][:,3]-s[1]["boxes"][:,1]) for s in samples ]
        widths = [ np.mean(s[1]["boxes"][:,2]-s[1]["boxes"][:,0]) for s in samples ]
        line_counts = [ len(s[1]['boxes']) for s in samples ]

        height_stats = [ int(s) for s in(np.mean( heights ), np.median(heights), np.min(heights), np.max(heights))]
        width_stats = [int(s) for s in (np.mean( widths ), np.median(widths), np.min(widths), np.max(widths))]
        line_count_stats = [int(s) for s in (np.mean( line_counts ), np.median(line_counts), np.min(line_counts), np.max(line_counts))]

        stat_list = ('Mean', 'Median', 'Min', 'Max')
        row_format = "{:>15}" * (len(stat_list) + 1)
        return '\n'.join([
            row_format.format("", *stat_list),
            row_format.format("Line height", *height_stats),
            row_format.format("Line width", *width_stats),
            row_format.format("Line count/page", *line_count_stats),
        ])

    def _generate_readme( self, filename: str, params: dict )->None:
        """Create a metadata file in the work directory.

        Args:
            filename (str): a filepath.
            params (dict): dictionary of parameters passed to the dataset task builder.

        Returns:
            None
        """
        filepath = Path(self.work_folder_path, filename )
        
        with open( filepath, "w") as of:
            print('Task was built with the following options:\n\n\t+ ' + 
                  '\n\t+ '.join( [ f"{k}={v}" for (k,v) in params.items() ] ),
                  file=of)
            print( repr(self), file=of)


    def _build_line_seg_samples(self, raw_data_folder_path:Path,
                                work_folder_path: Path, ) -> List[Tuple[Path, Dict[str,Tensor]]]:
        """Create a new dataset for segmentation that associate each page image with its metadata.

        Args:
            raw_data_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder (relative to the caller's pwd).
            on_disk (bool): If False (default), samples are only built in memory; otherwise line images 
                and metadata are written into the work folder.

        Returns:
            Tuple[Path, List[dict]]: a list of pairs `(<absolute img filepath>, <absolute transcription filepath>)`
        """
        Path( work_folder_path ).mkdir(exist_ok=True, parents=True) # always create the subfolder if not already there
        if not self.config['resume_task']:
            self._purge( work_folder_path ) 

        cnt = 0
        samples = []

        for page in tqdm(self.pages):
            with open(page, 'r') as page_file:
                page_id = re.match(r'(.+).{}'.format(self.config['gt_suffix']), page.name).group(1)
                img_path, sample_img_path=None, None
                for img_suffix in ('.jpg', '.png', '.img.jpg'):
                    img_path = Path(self.raw_data_folder_path, f'{page_id}{img_suffix}' )
                    if img_path.exists():
                        page_id = page_id.replace('.', '_')
                        #img_path.hardlink_to( work_folder_path.joinpath( f'{page_id}{img_suffix}' ))
                        sample_img_path = work_folder_path.joinpath( f'{page_id}{img_suffix}' ).absolute()
                        os.link(img_path, sample_img_path)
                        break
                if not img_path.exists():
                    raise FileNotFoundError(f"Could not find a valid image file with prefix {page_id}")

                # extract metadata
                boxes, masks = None, None
                if self.config['gt_suffix'] == 'xml':
                    boxes, masks = seglib.line_masks_from_img_xml_files( img_path, page )
                elif 'json' in self.config['gt_suffix']:
                    boxes, masks = seglib.line_masks_from_img_json_files( img_path, page, key=self.config['polygon_key'] )

                sample = (sample_img_path, { 'boxes': boxes, 'masks': masks })
                # - on-disk: 1 image + 1 tensor of boxes + 1 tensor of masks
                try:
                    page_image = Image.open( img_path, 'r')
                except Image.DecompressionBombWarning as dcb:
                    logger.debug( f'{dcb}: ignoring page' )
                    return None
                msk_filename = f'{page_id}.masks.npy.gz'
                with gzip.GzipFile( work_folder_path.joinpath(msk_filename), 'w') as zf:
                    np.save( zf, masks )
                box_filename = f'{page_id}.boxes.npy.gz'
                with gzip.GzipFile( work_folder_path.joinpath(box_filename), 'w') as zf:
                    np.save( zf, boxes )

                samples.append( sample )
                cnt += 1
                if self.config['count'] and cnt == self.config['count']:
                    break
        return samples


    @staticmethod
    def dump_data_to_tsv(samples: List[dict], file_path: str='', all_path_style=False) -> None:
        """Create a CSV file with all tuples (`<line image absolute path>`, `<boxes_tensor_file>`, `<masks_tensor_file>`)

        Args:
            samples (List[dict]): dataset samples.
            file_path (str): A TSV (absolute) file path (Default value = '')

        Returns:
            None
        """
        if not file_path:
            return
        with open( file_path, 'w') as of:
            for sample in samples:
                img_filename = sample[0].name
                work_folder_path = sample[0].parent
                print(work_folder_path)
                file_prefix = re.sub(r'\..+', '', img_filename )
                boxes_filename = Path(file_prefix).with_suffix('.boxes.npy.gz')
                masks_filename = Path(file_prefix).with_suffix('.masks.npy.gz')
                print(work_folder_path.joinpath(masks_filename))
                print(work_folder_path.joinpath(boxes_filename))
                if not work_folder_path.joinpath(boxes_filename).exists():
                    raise FileNotFoundError("Could not find {}. Abort.".format( boxes_filename ))
                if not work_folder_path.joinpath(masks_filename).exists():
                    raise FileNotFoundError("Could not find {}. Abort.".format( masks_filename ))
                of.write("{}\t{}\t{}".format( img_filename, boxes_filename, masks_filename ))


    @staticmethod
    def _split_set(samples: object, ratios: Tuple[float, float, float], subset: str) -> List[object]:
        """Split a dataset into 3 sets: train, validation, test.

        Args:
            samples (object): any dataset sample.
            ratios (Tuple[float, float, float]): respective proportions for possible subsets
            subset (str): subset to be build  ('train', 'validate', or 'test')

        Returns:
            List[object]: a list of samples.

        Raises:
            ValueError: The subset type does not exist.
        """

        random.seed(10)
        logger.debug("Splitting set of {} samples with ratios {}".format( len(samples), ratios))

        if 1.0 in ratios:
            return list( samples )
        if subset not in ('train', 'validate', 'test'):
            raise ValueError("Incorrect subset type: choose among 'train', 'validate', and 'test'.")

        subset_2_count = int( len(samples)* ratios[1])
        subset_3_count = int( len(samples)* ratios[2] )

        subset_1_indices = set( range(len(samples)))
        
        if ratios[1] != 0:
            subset_2_indices = set( random.sample( subset_1_indices, subset_2_count))
            subset_1_indices -= subset_2_indices

        if ratios[2] != 0:
            subset_3_indices = set( random.sample( subset_1_indices, subset_3_count))
            subset_1_indices -= subset_3_indices

        if subset == 'train':
            return [ samples[i] for i in subset_1_indices ]
        if subset == 'validate':
            return [ samples[i] for i in subset_2_indices ]
        if subset == 'test':
            return [ samples[i] for i in subset_3_indices ]


    def __getitem__(self, index) -> Dict[str, Union[Tensor, int, str]]:
        """Callback function for the iterator.

        Args:
            index (int): item index.

        Returns:
            dict[str,Union[Tensor,int,str]]: a tuple with
                + the input image tensor (C,H,W)
                + the segmentation metadata dictionary with tensor 
                  of boxes (N,4) and tensor of masks (N,H,W)
        """
        img_path = self.data[index][0]
        assert isinstance(img_path, Path) or isinstance(img_path, str)

        # In the sample, image filename replaced with 
        # - file id ('id')
        # - tensor ('img')
        # - dimensions of transformed image ('height' and 'width')
        # 
        sample = self.data[index]
        img_array_hwc = ski.io.imread( img_path )
        
        return (
            v2.Compose( [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img_array_hwc),
            { 
                'id': Path(img_path).name,
                'boxes': torch.tensor(sample[1]['boxes']),
                'labels': torch.ones( (len(sample[1]['boxes'])), dtype=torch.int64),
                'masks': torch.tensor( sample[1]['masks']), 
             })


    def __getitems__(self, indexes: list ) -> List[dict]:
        """To help with batching.

        Args:
            indexes (list): a list of indexes.

        Returns:
            List[dict]: a list of samples.
        """
        return [ self.__getitem__( idx ) for idx in indexes ]


    def __len__(self) -> int:
        """Number of samples in the dataset.

        Returns:
            int: number of data points.
        """
        return len( self.data )


    def _purge(self, folder: str) -> int:
        """Empty the line image subfolder: all line images and transcriptions are
        deleted, as well as the TSV file.

        Args:
            folder (str): Name of the subfolder to _purge (relative the caller's pwd

        Returns:
            int: number of deleted files.
        """
        cnt = 0
        for item in [ f for f in Path( folder ).iterdir() if not f.is_dir()]:
            item.unlink()
            cnt += 1
        return cnt

    def __repr__(self) -> str:

        summary = '\n'.join([
                    f"Root folder:\t{self.root}",
                    f"Files extracted in:\t{self.raw_data_folder_path}",
                    f"Work folder:\t{self.work_folder_path}",
                    f"Data points:\t{len(self.data)}",
                    "Stats:",
                    f"{self.dataset_stats(self.data)}" if self.data else 'No data',])

        return ("\n________________________________\n"
                f"\n{summary}"
                "\n________________________________\n")



class MonasteriumDataset(ChartersDataset):
    """A subset of Monasterium charter images and their meta-data (PageXML).

        + its core is a set of charters segmented and transcribed by various contributors, mostly by correcting Transkribus-generated data.
        + it has vocation to grow through in-house, DiDip-produced transcriptions.
    """

    dataset_resource = {
            #'url': r'https://cloud.uni-graz.at/apps/files/?dir=/DiDip%20\(2\)/CV/datasets&fileid=147916877',
            'url': r'https://drive.google.com/uc?id=1hEyAMfDEtG0Gu7NMT7Yltk_BAxKy_Q4_',
            'tarball_filename': 'MonasteriumTekliaGTDataset.tar.gz',
            'md5': '7d3974eb45b2279f340cc9b18a53b47a',
            'full-md5': 'e720bac1040523380921a576f4cc89dc',
            'desc': 'Monasterium ground truth data (Teklia)',
            'origin': 'google',
            'tarball_root_name': 'MonasteriumTekliaGTDataset',
            'comment': 'A clean, terse dataset, with no use of Unicode abbreviation marks.',
    }

    work_folder_name="MonasteriumHandwritingDataset"

    root_folder_basename="Monasterium"

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)


class KoenigsfeldenDataset(ChartersDataset):
    """A subset of charters from the Koenigsfelden abbey, covering a wide range of handwriting style.
        The data have been compiled from raw Transkribus exports.
    """

    dataset_resource = {
            'file': f"{os.getenv('HOME')}/tmp/data/koenigsfelden_abbey_1308-1662/koenigsfelden_1308-1662.tar.gz",
            'tarball_filename': 'koenigsfelden_1308-1662.tar.gz',
            'md5': '9326bc99f9035fb697e1b3f552748640',
            'desc': 'Koenigsfelden ground truth data',
            'origin': 'local',
            'tarball_root_name': 'koenigsfelden_1308-1662',
            'comment': 'Transcriptions have been cleaned up (removal of obvious junk or non-printable characters, as well a redundant punctuation marks---star-shaped unicode symbols); unicode-abbreviation marks have been expanded.',
    }

    work_folder_name="KoenigsfeldenHandwritingDataset"
    "This prefix will be used when creating a work folder."

    root_folder_basename="Koenigsfelden"
    "This is the root of the archive tree."

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)

        #self.target_transform = self.filter_transcription




class KoenigsfeldenDatasetAbbrev(ChartersDataset):
    """A subset of charters from the Koenigsfelden abbey, covering a wide range of handwriting style.
        The data have been compiled from raw Transkribus exports.
    """

    dataset_resource = {
            'file': f"{os.getenv('HOME')}/tmp/data/koenigsfelden_abbey_1308-1662/koenigsfelden_1308-1662.tar.gz",
            'tarball_filename': 'koenigsfelden_1308-1662_abbrev.tar.gz',
            'md5': '9326bc99f9035fb697e1b3f552748640',
            'desc': 'Koenigsfelden ground truth data',
            'origin': 'local',
            'tarball_root_name': 'koenigsfelden_1308-1662_abbrev',
            'comment': 'Similar to the KoenigsfeldenDataset, with a notable difference: Unicode abbreviations have been kept.',
    }

    work_folder_name="KoenigsfeldenHandwritingDataset"
    "This prefix will be used when creating a work folder."

    root_folder_basename="KoenigsfeldenAbbrev"
    "This is the root of the archive tree."

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)

        #self.target_transform = self.filter_transcription


class NurembergLetterbooks(ChartersDataset):
    """
    Nuremberg letterbooks (15th century).
    """

    dataset_resource = {
            'file': f"{os.getenv('HOME')}/tmp/data/nuremberg_letterbooks/nuremberg_letterbooks.tar.gz",
            'tarball_filename': 'nuremberg_letterbooks.tar.gz',
            'md5': '9326bc99f9035fb697e1b3f552748640',
            'desc': 'Nuremberg letterbooks ground truth data',
            'origin': 'local',
            'tarball_root_name': 'nuremberg_letterbooks',
            'comment': 'Numerous struck-through lines (masked)'
    }

    work_folder_name="NurembergLetterbooksDataset"
    "This prefix will be used when creating a work folder."

    root_folder_basename="NurembergLetterbooks"
    "This is the root of the archive tree."

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)




def dummy():
    """"""
    return True
