import numpy as np
from easydict import EasyDict as edict

config = edict()
config.root = 'F:/train-mtcnn-head'

config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = True
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.enable_gray = True
config.enable_blur = True
