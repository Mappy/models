from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.DATA_FACTORY = '/data/'
cfg.DATA_IMDB = '/data/training_1388/'

cfg.OUTPUT_DIR = '/data/training_snapshots/'

# Load a specific tfmodel
cfg.LOAD_TFMODEL = True
# Path to the specific tfmodel
cfg.TFMODEL = '/data/voc_2007_trainval+voc_2012_trainval/res101_faster_rcnn_iter_110000.ckpt'
# cfg.TFMODEL = '/data/voc_2007_trainval+voc_2012_trainval/vgg16_faster_rcnn_iter_110000.ckpt'

# Connect bo-pano
cfg.BO = edict()
cfg.BO.id = "IA"
cfg.BO.password = "mappyIaBoss"
cfg.BO.address_port = "snap-panoramicbo-003.mappy.priv:9090"
#cfg.BO.address_port = "10.20.0.168:9090"

# Combining shape
cfg.NB_BOX_W = 100
cfg.MIN_IOU = 0.001

#
# Training options
#
cfg.TRAIN = edict()

cfg.TRAIN.NB_EXAMPLE = 1300
cfg.TRAIN.MODE = 'bbox'
cfg.TRAIN.MODE = 'all'

cfg.TRAIN.DISPLAY_INFO = 1

# Initial learning rate
cfg.TRAIN.LEARNING_RATE = 0.001
cfg.TRAIN.LEARNING_RATE = 0.001

# Momentum
cfg.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
cfg.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
cfg.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
cfg.TRAIN.STEPSIZE = 5000

# Iteration intervals for showing the loss during training, on command line interface
cfg.TRAIN.DISPLAY = 5

# Whether to double the learning rate for bias
cfg.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution
cfg.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
cfg.TRAIN.BIAS_DECAY = False

# Whether to add ground truth boxes to the pool when sampling regions
cfg.TRAIN.USE_GT = False

# Whether to use aspect-ratio grouping of training images, introduced merely for saving
# GPU memory
cfg.TRAIN.ASPECT_GROUPING = False

# The number of snapshots kept, older ones are deleted to save space
cfg.TRAIN.SNAPSHOT_KEPT = 250

# The time interval for saving tensorflow summaries
cfg.TRAIN.SUMMARY_INTERVAL = 10

# The scale is the pixel size of an image's shortest side
cfg.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
cfg.TRAIN.MAX_SIZE = 3072

# Images to use per minibatch
cfg.TRAIN.IMS_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
cfg.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
cfg.TRAIN.FG_FRACTION = 0.25
cfg.TRAIN.FG_FRACTION = 0.85

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
cfg.TRAIN.FG_THRESH = 0.5
cfg.TRAIN.FG_THRESH = 0.01

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
cfg.TRAIN.BG_THRESH_HI = 0.5
cfg.TRAIN.BG_THRESH_LO = 0.1

cfg.TRAIN.BG_THRESH_HI = 0.01
cfg.TRAIN.BG_THRESH_LO = 0.0

# Use horizontally-flipped images during training?
cfg.TRAIN.USE_FLIPPED = False

# Train bounding-box regressors
cfg.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
cfg.TRAIN.BBOX_THRESH = 0.5
cfg.TRAIN.BBOX_THRESH = 0.05

# Iterations between snapshots
cfg.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
# __C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'
cfg.TRAIN.SNAPSHOT_PREFIX = 'vgg16_allnet_train'
# __C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
# __C.TRAIN.USE_PREFETCH = False

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
cfg.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
cfg.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
cfg.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
cfg.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
cfg.TRAIN.PROPOSAL_METHOD = 'gt'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.

# Use RPN to detect objects
cfg.TRAIN.HAS_RPN = True
# IOU >= thresh: positive example
cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.1

# IOU < thresh: negative example
cfg.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
cfg.TRAIN.RPN_NEGATIVE_OVERLAP = 0.05

# If an anchor statisfied by positive and negative conditions set to negative
cfg.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
cfg.TRAIN.RPN_FG_FRACTION = 0.5
cfg.TRAIN.RPN_FG_FRACTION = 0.90
# Total number of examples
cfg.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
cfg.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
cfg.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
cfg.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
# __C.TRAIN.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
cfg.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
# Whether to use all ground truth bounding boxes for training,
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
cfg.TRAIN.USE_ALL_GT = True

#
# Testing options
#
cfg.TEST = edict()

cfg.TEST.NB_EXAMPLE = 10

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
cfg.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
cfg.TEST.MAX_SIZE = 3072

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
cfg.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
cfg.TEST.SVM = False

# Test using bounding-box regressors
cfg.TEST.BBOX_REG = True

# Propose boxes
cfg.TEST.HAS_RPN = False

# Test using these proposals
cfg.TEST.PROPOSAL_METHOD = 'gt'

## NMS threshold used on RPN proposals
cfg.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
# __C.TEST.RPN_PRE_NMS_TOP_N = 60000

## Number of top scoring boxes to keep after applying NMS to RPN proposals
cfg.TEST.RPN_POST_NMS_TOP_N = 300
# __C.TEST.RPN_POST_NMS_TOP_N = 3000

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
# __C.TEST.RPN_MIN_SIZE = 16

# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
cfg.TEST.MODE = 'nms'

# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
cfg.TEST.RPN_TOP_N = 5000

#
# ResNet options
#

cfg.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize.
# if true, the region will be resized to a squre of 2xPOOLING_SIZE,
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
cfg.RESNET.MAX_POOL = False

# Number of fixed blocks during finetuning, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
cfg.RESNET.FIXED_BLOCKS = 1

# Whether to tune the batch nomalization parameters during training
cfg.RESNET.BN_TRAIN = False

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
cfg.DEDUP_BOXES = 1. / 16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
cfg.RNG_SEED = 3

# A small number that's used many times
cfg.EPS = 1e-14

# Root directory of project
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
cfg.DATA_DIR = osp.abspath(osp.join(cfg.ROOT_DIR, 'data'))

# Name (or path to) the matlab executable
cfg.MATLAB = 'matlab'

# Place outputs under an experiments directory
cfg.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
cfg.USE_GPU_NMS = True

# Default GPU device id
cfg.GPU_ID = 0

# Default pooling mode, only 'crop' is available
cfg.POOLING_MODE = 'crop'

# Size of the pooled region after RoI pooling
cfg.POOLING_SIZE = 7

# Anchor scales for RPN
cfg.ANCHOR_SCALES = [8, 16, 32]
# __C.ANCHOR_SCALES = [2,4,8,16,32]

# Anchor ratios for RPN
cfg.ANCHOR_RATIOS = [0.5, 1, 2]


def get_output_dir(imdb, weights_filename):
    return cfg.OUTPUT_DIR


def get_output_tb_dir(imdb, weights_filename):
    """Return the directory where tensorflow summaries are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(cfg.ROOT_DIR, 'tensorboard', cfg.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, cfg)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = cfg
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
