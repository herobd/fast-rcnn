#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='CUB_2011_test', type=str)
    parser.add_argument('--res', dest='res_file',
                        help='pickle of all_boxes',
                        default='/home/brianld/fast-rcnn/data/CUB_200_2011/results/all_boxes_CUB2011.pkl', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)



    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    imdb.evaluate_detections_from(args.res_file)
