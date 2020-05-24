from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# takes file path as input, returns list of blocks
def parse_cfg(cfgfile):
    # save content of cfg file as list of strings
    file = open(cfgfile, 'r')
    # split into lines
    content = file.read().split('\n')
    # remove empty lines
    content = [line for line in content if len(line) > 0]
    # remove any comments
    content = [line for line in content if line[0] != '#']
    # remove any extra spaces
    content = [line.rstrip().lstrip() for line in content]

    block, blocks = {}, []
    for line in content:
        # start of a new block
        if line[0] == ''['':
            # if not empty, save data from old block and start fresh
            if len(block) != 0:
                blocks.append(block)
                block = {}
            # otherwise, save type of block (without [])
            block['type'] = line[1:-1].rstrip()
        else:
            # store attributes as key value pair
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    # when EOF need to save final block
    blocks.append(block)

    return blocks

def create_modules(blocks):
    met_info = blocks[0]
    module_list = nn.ModuleList()
    # must keep track of number of filters in previous layer
    # init as 3 (RGB)
    prev_filters = 3
    output_filters = []

    # iterate over list of blocks, create pytorch module
    # skip first block since it only holds net info
    for index, block in enumerate(blocks[1:]):
        # use this class to execute a number of nn.Module objects
        module = nn.Sequential()

        # create convolutional block (3 pieces)
        if block['type'] == 'convolutional':
            # collect block information
            activation = block['activation']
            try:
                batch_normalize = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(block['filters'])
            padding = int(block['padding'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            # add convolutional layer
            convolution_layer = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), convolution_layer)
            # add batch norm layer
            if batch_normalize:
                batch_norm_layer = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), batch_norm_layer)
            # add linear or leaky reLu activation layer (check first)
            if activation == 'leaky':
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activation_layer)

        # create upsampling block
        elif (block['type'] == 'upsample'):
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{0}'.format(index), upsample)
