import sys
from utils.shading import *

# other modules
import os
import numpy as np

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2
from models.dpr.defineHourglass_1024_gray_skip_matchFeature import *

def get_dpr_network():
    modelFolder = 'trained_model/'

    # load model

    my_network_512 = HourglassNet(16)
    my_network = HourglassNet_1024(my_network_512, 16)
    my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_1024_03.t7')))
    my_network.cuda()
    my_network.train(False)


    return my_network