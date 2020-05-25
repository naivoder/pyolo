"""
this file parses a cfg input and builds the underlying architecture of the yolo network

"""

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *

# dummy function to test forward pass
def test_input():
    img = cv2.imread('test.png')
    # resize to standard input
    img = cv2.resize(img, (416, 416))
    # BGR -> RGB | HxWC -> CxHxW
    img = img[:, :, ::-1].transpose((2, 0, 1))
    # add 4th channel for batch and normalize
    img = img[np.newaxis, :, :, :] / 255.0
    # convert pixel values to floats
    img = torch.from_numpy(img).float()
    # convert to autograd variable
    img = Variable(img)
    return img

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
        if line[0] == '[':
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

# custom layer for detecting bounding boxes
# https://pytorch.org/docs/stable/nn.html#torch.nn.Module
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

# custom empty layer
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

def create_modules(blocks):
    net_info = blocks[0]
    # https://pytorch.org/docs/stable/nn.html#torch.nn.ModuleList
    module_list = nn.ModuleList()
    # must keep track of number of filters in previous layer
    # init as 3 (RGB)
    prev_filters = 3
    output_filters = []

    # iterate over list of blocks, create pytorch module
    # skip first block since it only holds net info
    for index, block in enumerate(blocks[1:]):
        # use this class to execute a number of nn.Module objects
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential
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
            padding = int(block['pad'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            # add convolutional layer
            # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
            convolution_layer = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), convolution_layer)
            # add batch norm layer
            if batch_normalize:
                # https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d
                batch_norm_layer = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), batch_norm_layer)
            # add linear or leaky reLu activation layer (check first)
            if activation == 'leaky':
                # https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activation_layer)

        # create upsampling block
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            # https://pytorch.org/docs/stable/nn.html#torch.nn.Upsample
            upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{0}'.format(index), upsample_layer)

        # create route block
        elif block['type'] == 'route':
            # split layers into list
            block['layers'] = block['layers'].split(',')
            # start of route
            start = int(block['layers'][0])
            # end of route
            try:
                end = int(block['layers'][1])
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route_layer = EmptyLayer()
            module.add_module('route_{0}'.format(index), route_layer)
            if end < 0:
                # concatenate feature maps with previous layer
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # create shortcut (skip connection) block
        elif block['type'] == 'shortcut':
            shortcut_layer = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index), shortcut_layer)

        # create yolo block -> detection layer
        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [int(m) for m in mask]
            anchors = block['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[m] for m in mask]
            detection_layer = DetectionLayer(anchors)
            module.add_module('detection_{0}'.format(index), detection_layer)

        # record information from loop
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

# custom network architecture definition
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    # override forward pass method of nn.Module class
    def forward(self, net, CUDA=True):
        # net info block is skipped
        modules = self.blocks[1:]
        # cache of output feature maps
        # write flag, 1 means collector initialized
        # i.e. have encountered first detection
        # we delay initializing collection as empty tensor
        outputs = {}; write = 0
        for index, module in enumerate(modules):
            module_type = module['type']

            # if convolutional or upsample
            if module_type == 'convolutional' or module_type == 'upsample':
                net = self.module_list[index](net)

            # if route layer
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(l) for l in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - index
                if len(layers) == 1:
                    net = outputs[index + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - index
                    map_1 = outputs[index + layers[0]]
                    map_2 = outputs[index + layers[1]]
                    net = torch.cat((map_1, map_2), 1)
                    
            # if shortcut layer
            elif module_type == 'shortcut':
                start = int(module['from'])
                net = outputs[index - 1] + outputs[index + start]
            # can concatenate detection maps at three scales
            # using predict_transform helper function
            # output tensor is now table with bounding boxes as rows
            elif module_type == 'yolo':
                anchors = self.module_list[index][0].anchors
                # retrieve input dimensions and class numbers
                shape = int(self.net_info['height'])
                num_classes = int(module['classes'])
                # perform output transform of prediction
                net = net.data
                net = predict_transform(net, shape, anchors, num_classes, CUDA)
                if not write:
                    detections = net
                    write = 1
                else:
                    detections = torch.cat((detections, net), 1)
            outputs[index] = net
        return detections

if __name__=="__main__":
    blocks = parse_cfg('cfg/yolov3.cfg')
    print(create_modules(blocks))
    model = Darknet('cfg/yolov3.cfg')
    net_input = test_input()
    net_output = model(net_input, torch.cuda.is_available())
    print(net_output)
