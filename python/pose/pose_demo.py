#!/usr/bin/env python
"""
Pose predictions in Python.

Caffe must be available on the Pythonpath for this to work. The methods can
be imported and used directly, or the command line interface can be used. In
the latter case, adjust the log-level to your needs. The maximum image size
for one prediction can be adjusted with the variable _MAX_SIZE so that it
still fits in GPU memory, all larger images are split in sufficiently small
parts.

Authors: Christoph Lassner, based on the MATLAB implementation by Eldar
  Insafutdinov.
"""
# pylint: disable=invalid-name
import os as _os
import logging as _logging
import glob as _glob
import numpy as _np
import scipy as _scipy
import click as _click
import caffe as _caffe
import random
import PIL
from estimate_pose import estimate_pose
from PIL import Image, ImageDraw
_LOGGER = _logging.getLogger(__name__)

def fake_npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = _np.ogrid[-radius: radius, -radius: radius]
    #cy = cy + 10*random.uniform(1, 2)
    #cx = cx + 10*random.uniform(1, 2)
    index = x**2 + y**2 <= radius**2
    image[cy-radius:cy+radius, cx-radius:cx+radius][index] = (
        image[cy-radius:cy+radius, cx-radius:cx+radius][index].astype('float32') * transparency +
        _np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')

def getline(image, pose ):
    line = [[0,1],[1,2],[2,12],[3,12],[3,4],[4,5],[12,13],[12,8],[8,7],[7,6],[12,9],[9,10],[10,11]]
    lst = []
    head =(pose[0, 13],pose[1, 13])
    neck =(pose[0, 12],pose[1, 12])
    left_hand = (pose[0, 6],pose[1, 6])
    left_arm = (pose[0, 7],pose[1, 7])
    left_shoulder =  (pose[0, 8],pose[1, 8])
    right_hand = (pose[0, 11],pose[1, 11])
    right_arm = (pose[0, 10],pose[1, 10])
    right_shoulder =  (pose[0, 9],pose[1, 9])
    left_foot = (pose[0, 0],pose[1, 0])
    left_leg = (pose[0, 1],pose[1, 1])
    left_kua =  (pose[0, 2],pose[1, 2])
    right_foot = (pose[0, 3],pose[1, 3])
    right_leg = (pose[0, 4],pose[1, 4])
    right_kua =  (pose[0, 5],pose[1, 5])
    
    im = PIL.Image.fromarray(_np.uint8(image))
    draw = ImageDraw.Draw(im) 
    for par in line:
        draw.line((pose[0][par[0]],pose[1][par[0]],pose[0][par[1]],pose[1][par[1]]), fill=128,width=3)
        
        #print(pose[0][par[0]],pose[1][par[0]],pose[0][par[1]],pose[1][par[1]])
    im.save('test.png')
    


def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = _np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    image[cy-radius:cy+radius, cx-radius:cx+radius][index] = (
        image[cy-radius:cy+radius, cx-radius:cx+radius][index].astype('float32') * transparency +
        _np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')


###############################################################################
# Command line interface.
###############################################################################

@_click.command()
@_click.argument('image_name',
                 type=_click.Path(exists=True, dir_okay=True, readable=True))
@_click.option('--out_name',
               type=_click.Path(dir_okay=True, writable=True),
               help='The result location to use. By default, use `image_name`_pose.npz.',
               default=None)
@_click.option('--scales',
               type=_click.STRING,
               help=('The scales to use, comma-separated. The most confident '
                     'will be stored. Default: 1.'),
               default='1.')
@_click.option('--visualize',
               type=_click.BOOL,
               help='Whether to create a visualization of the pose. Default: True.',
               default=True)
@_click.option('--folder_image_suffix',
               type=_click.STRING,
               help=('The ending to use for the images to read, if a folder is '
                     'specified. Default: .png.'),
               default='.png')
@_click.option('--use_cpu',
               type=_click.BOOL,
               is_flag=True,
               help='Use CPU instead of GPU for predictions.',
               default=False)
@_click.option('--gpu',
               type=_click.INT,
               help='GPU device id.',
               default=0)
def predict_pose_from(image_name,
                      out_name=None,
                      scales='1.',
                      visualize=True,
                      folder_image_suffix='.png',
                      use_cpu=False,
                      gpu=0):
    """
    Load an image file, predict the pose and write it out.
    
    `IMAGE_NAME` may be an image or a directory, for which all images with
    `folder_image_suffix` will be processed.
    """
    model_def = '../../models/deepercut/ResNet-152.prototxt'
    model_bin = '../../models/deepercut/ResNet-152.caffemodel'
    #model_def = '/home/bnrc2/deepcut-pose/deepercut/ResNet-101-deploy.prototxt'
    #model_bin = '/home/bnrc2/deepcut-pose/deepercut/finetune_train_iter_1030000.caffemodel'

    scales = [float(val) for val in scales.split(',')]
    if _os.path.isdir(image_name):
        folder_name = image_name[:]
        _LOGGER.info("Specified image name is a folder. Processing all images "
                     "with suffix %s.", folder_image_suffix)
        images = _glob.glob(_os.path.join(folder_name, '*' + folder_image_suffix))
        process_folder = True
    else:
        images = [image_name]
        process_folder = False
    if use_cpu:
        _caffe.set_mode_cpu()
    else:
        _caffe.set_mode_gpu()
        _caffe.set_device(gpu)
    out_name_provided = out_name
    if process_folder and out_name is not None and not _os.path.exists(out_name):
        _os.mkdir(out_name)
    for image_name in images:
        if out_name_provided is None:
            out_name = image_name + '_pose.npz'
        elif process_folder:
            out_name = _os.path.join(out_name_provided,
                                     _os.path.basename(image_name) + '_pose.npz')
        _LOGGER.info("Predicting the pose on `%s` (saving to `%s`) in best of "
                     "scales %s.", image_name, out_name, scales)
        image = _scipy.misc.imread(image_name)
        if image.ndim == 2:
            _LOGGER.warn("The image is grayscale! This may deteriorate performance!")
            image = _np.dstack((image, image, image))
        else:
            image = image[:, :, ::-1]    
        pose = estimate_pose(image, model_def, model_bin, scales)
        _np.savez_compressed(out_name, pose=pose)
        print(pose)
        if visualize:
            visim = image[:, :, ::-1].copy()
            colors = [[255, 0, 0],[0, 255, 0],[0, 0, 255],[255,128,0],[255,0,128],[255,255,0],
                      [255, 0, 255],[0, 255, 255],[128, 128, 128],[128,0,128],[0,128,128],[128,128,0],
                      [0,0,0],[255,255,255]]
            for p_idx in range(14):
                _npcircle(visim,
                          pose[0, p_idx],
                          pose[1, p_idx],
                          8,
                          colors[p_idx],
                          0.0)

        vis_name = out_name + '_vis.png'
        #_scipy.misc.imsave(vis_name, visim)
        getline(visim,pose)
        

if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    # pylint: disable=no-value-for-parameter
    predict_pose_from()
