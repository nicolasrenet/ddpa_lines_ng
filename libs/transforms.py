# nprenet@gmail.com
# 05.2025

"""
Unused transforms, for reference. (This project uses Tormentor instead.)
"""



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

    Example:


    train_transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize( hyper_params['img_size'] ),
            RandomElasticGrid(p=0.3, grid_cols=(4,20)),
            v2.RandomRotation( 5 ),
            v2.ToDtype(torch.float32, scale=True),
            ])

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



