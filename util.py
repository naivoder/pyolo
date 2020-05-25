from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

# output of yolo is convolutional feature map
# not very convenient for output processing
# detections also occur at three different scales
# -> helper function to process yolo output
# takes feature map as input, returns 2D tensor
# each row corresponds to attribute of bounding box
def predict_transform(prediction, shape, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = shape // prediction.size(2)
    grid_size = shape // stride
    bbox_attributes = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attributes * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attributes)
    # dimensions of anchors correspond to height and width in net block
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    # sigmoid 'squishify' function for (x, y, confidence)
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    # add grid offsets to center coordinate position
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    # if CUDA:
    #     x_offset = x_offset.cuda()
    #     y_offset = y_offset.cuda()
    # ...what the heck is this section doing??
    # we're concatenating two offset meshgrids, what are they?
    # what are the calls to 'view' accomplishing? what is 'unsqueeze'?
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    # apply offset to prediction
    prediction[:, :, :2] += x_y_offset
    # apply anchors to dimensions of bounding box via log space transform
    anchors = torch.FloatTensor(anchors)
    # if CUDA:
    #     anchors = anchors.cuda()
    # what are the results of these transformations?
    # need to add debug clause and print shapes at each step...
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    # apply sigmoid activation to the class scores
    prediction[:, :, 5:5 + num_classes] = torch.sigmoid((prediction[:, :, 5:5 + num_classes]))
    # finally, resize detection map to the size of the input image
    # bounding box atttributes are sized according to feature map (eg 13, 13)
    # if input image is 416, 416 multiply by 32 -> 'stride' variable
    prediction[:, :, :4] *= stride
    return prediction
