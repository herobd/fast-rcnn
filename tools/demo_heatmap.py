#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

from datasets.factory import list_imdbs
import datasets
import subprocess

"""CLASSES = ('__background__',
           'aeroplane', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
"""
CLASSES = ('__background__', # always index 0
                         'airplane', 'bicycle', 'bird','cat')
#self._class_ids = ('__background__', # always index 0
#                        'n02691156', 'n02834778', 'n01503061','n02121620')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel_backup'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel_backup')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print 'error, no good inds'
        print dets
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'imagenetdevkit2014','imagenet2014', 'JPEGImages_test', image_name + '.JPEG')
    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                            image_name + '_boxes.mat')
    
    cmd = 'cd {} && '.format('/home/brianld/fast-rcnn/lib/datasets/')
    cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'do_selective_search_for_im(\'{:s}\',\'{:s}\'); quit;"' \
             .format(im_file, box_file)
    print('Running:\n{}'.format(cmd))
    status = subprocess.call(cmd, shell=True)
    assert os.path.exists(box_file), \
             'Selective search data not found at: {}'.format(box_file)
    print 'ROI done loading'
    raw_data = sio.loadmat(box_file)['boxes']
    obj_proposals=(raw_data[:, (1, 0, 3, 2)] - 1)
    #print obj_proposals
    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    #print scores
    #print boxes
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        #print cls_boxes
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        #print cls_boxes
        for ii in xrange(cls_boxes.shape[0]):
            assert cls_boxes[ii,2]-cls_boxes[ii,0] >0
            assert cls_boxes[ii,3]-cls_boxes[ii,1] >0

        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                    CONF_THRESH)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
        
        #Brian's code{
        
        heatmap = np.zeros(im.shape, np.uint8)
        for i in range(0,len(cls_scores)):
            #centerX = cls_boxes[i][0] + (cls_boxes[i][2]-cls_boxes[i][0])/2
            #centerY = cls_boxes[i][1] + (cls_boxes[i][3]-cls_boxes[i][1])/2
            #heatmap[centerY-1:centerY+1,centerX-1:centerX+1]=(255*cls_scores[i],255*cls_scores[i],255*cls_scores[i])
            pix_val=255*cls_scores[i]
            for x in range(int(cls_boxes[i][0]),int(cls_boxes[i][2])+1):
               for y in range(int(cls_boxes[i][1]),int(cls_boxes[i][3])+1): 
                   (b,g,r)=heatmap[y,x]
                   if b<pix_val:
                       heatmap[y,x]=(pix_val,pix_val,pix_val)
        #}
        
        cv2.imwrite(os.path.join(cfg.ROOT_DIR,'output','heatmap_'+image_name+'_'+cls+'.png'),heatmap)
        #cv2.imshow('heatmap',heatmap)
        #cv2.waitKey()
        
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    caffemodel = os.path.join(cfg.ROOT_DIR, 'output','default','imagenet_2014_train', NETS[args.demo_net][1])
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],'test_5.prototxt')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Demo for n02691156_3785'
    demo(net, 'n02691156_12011', ('airplane',))
    demo(net, 'n02121620_945', ('cat',))
    demo(net, 'n01503061_13415', ('bird',))
    demo(net, 'n02834778_1709', ('bicycle',))

    #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #print 'Demo for data/demo/001551.jpg'
    #demo(net, '001551', ('sofa', 'tvmonitor'))

    plt.show()
    #print list_imdbs()
