# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.my_imagenet
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import re
import urllib
from multiprocessing import Pool

import cv2
import Image
import pylab as pl

def _worker_check((selfie,i)):
                path=selfie.image_path_at(i)
                if (path==''):
                    selfie.new_image_index[i]=''

class my_imagenet(datasets.imdb):
    def __init__(self, image_set, year, devkit_path=None):
        datasets.imdb.__init__(self, 'imagenet_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'imagenet' + self._year)
        """self._classes = ('__background__', # always index 0              TODO: examine this
                         'airplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        """
        #self._classes = (str(det_id) for det_id in range(0,200))
        """self._classes = ('__background__', # always index 0              TODO: examine this
                         '2', '24', '26', '4',
                         '5', '6', '27', '8', '9',
                         '10', '11', '12', '13',
                         '14', '15', '16', #ehhh no potted plant in imagenet
                         '17', '18', '19', '20')
        """
        self._classes = ('__background__', # always index 0
                         '2', '24', '26','28')
        self._class_ids = ('__background__', # always index 0
                         'n02691156', 'n02834778', 'n01503061','n02121620')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_id_to_ind = dict(zip(self._class_ids, xrange(self.num_classes)))
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()

        #Not all iamges are loaded, check. If we can't load them, strike them from the list
        self.new_image_index=self._image_index
        i=0
        while i<len(self._image_index):
            self.image_path_at(i)
            i+=1
        self._save_image_set_index()
        """
        nthreads = 20
        #pool = ThreadPool(processes=nthreads)
        pool = Pool(nthreads)



        pool.map(_worker_check, [(self,i) for i in range(1,len(self._image_index))])
        self._image_index = [iidx for iidx in new_image_index if (iidx!='')]
        self._save_image_set_index()
        """
        print 'Images done loading.'
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'imagenetdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        #print 'at '+str(i)
        #print self._image_index[i]
        toreturn= self.image_path_from_index(self._image_index[i])
        if toreturn != '':
            return toreturn
        
        del self._image_index[i]
        return self.image_path_at(i)

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages_'+self._image_set,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        """if not os.path.exists(image_path):
            f = open(os.path.join(self._devkit_path,'fall11_urls.txt'))
            find = re.compile('.+_'+re.sub(r'\w+_\w+_0*','',index)+'\s+(http.+)')
            for x in f.readlines():
                m=find.match(x)
                if m:
                    url=m.group(1)
                    print 'getting '+image_path+' from '+url
                    try:
                        urllib.urlretrieve(url,image_path)
                    except IOError:
                        print 'load failed'
                        return ''
                    break;
            f.close()
            if not os.path.exists(image_path):
                return ''
            #assert os.path.exists(image_path), \
            #        'Unable to download image for: {}'.format(image_path)
        """
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
    
    def _save_image_set_index(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '_new.txt')
        f = open(image_set_file,'w')
        for iidx in self._image_index:
            f.write(iidx+'\n')
        f.close()
        

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'imagenetdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            self._check_roidb(roidb)
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        self._check_roidb(gt_roidb)
        
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _check_roidb(self, roidb):
        ii=0
        for test in roidb:
            assert len(test['gt_classes'])>0, 'No gt_classes!'
            for cls in test['gt_classes']:
                assert cls>0 and cls<5, 'bg cls is ' +str(cls)+ ' at index: '+str(ii)
            for bb in test['boxes']:
                assert (bb[0]<=bb[2] and bb[1]<=bb[3]), 'bad bb at index: '+str(ii)
                assert not (bb[0]==0 and bb[2]==0), 'blank bb at index: '+str(ii)
            ii+=1

        num_images = self.num_images
        widths = [Image.open(self.image_path_at(i)).size[0]
                  for i in xrange(num_images)]
        for i in xrange(num_images):
            boxes = roidb[i]['boxes'].copy()
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            #image=cv2.imread(self.image_path_at(i))
            #width = image.shape[1]
            #height = image.shape[0]
            for ii in xrange(boxes.shape[0]):
                assert (boxes[ii, 2] < widths[i]), '[check] fail at '+str(ii)

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
       
            num_images = self.num_images
            widths = [Image.open(self.image_path_at(i)).size[0]
                  for i in xrange(num_images)]
            for i in xrange(num_images):
             boxes = roidb[i]['boxes'].copy()
             assert (boxes[:, 2] >= boxes[:, 0]).all()
             #image=cv2.imread(self.image_path_at(i))
             #width = image.shape[1]
             #height = image.shape[0]
             for ii in xrange(boxes.shape[0]):
                assert (boxes[ii, 2] < widths[i]), '[just loaded] fail at '+str(ii)

            return roidb

        if int(self._year) == 2014 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            self._check_roidb(gt_roidb)
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            self._check_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        
        #roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        num_images = self.num_images
        widths = [Image.open(self.image_path_at(i)).size[0]
                  for i in xrange(num_images)]
        for i in xrange(num_images):
            boxes = roidb[i]['boxes'].copy()
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            #image=cv2.imread(self.image_path_at(i))
            #width = image.shape[1]
            #height = image.shape[0]
            for ii in xrange(boxes.shape[0]):
                assert (boxes[ii, 2] < widths[i]), '[load] fail at '+str(ii)


        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        if not os.path.exists(filename):
            image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
            assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
            image_loc = os.path.join(self._data_path, 'JPEGImages')
            image_loc += '_'+self._image_set
            cmd = 'cd {} && '.format('/home/brianld/fast-rcnn/lib/datasets/')
            cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
            cmd += '-r "dbstop if error; '
            cmd += 'do_selective_search(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
                   .format(image_set_file, image_loc,self._image_ext, filename)
            print('Running:\n{}'.format(cmd))
            status = subprocess.call(cmd, shell=True)
            assert os.path.exists(filename), \
                   'Selective search data not found at: {}'.format(filename)
            print 'ROI done loading'
            
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
        for box in box_list:
            assert (box[:,2] >= box[:,0]).all()
            assert (box[:,0]>=0).all() and (box[:,1]>=0).all() and (box[:,2]>=0).all() and (box[:,3]>=0).all()
        assert len(box_list) >0, 'no bbs...'
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        assert False, 'not implemented'
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)


        num_images = self.num_images
        widths = [Image.open(self.image_path_at(i)).size[0]
                  for i in xrange(num_images)]
        for i in xrange(num_images):
            boxes = roidb[i]['boxes'].copy()
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            #image=cv2.imread(self.image_path_at(i))
            #width = image.shape[1]
            #height = image.shape[0]
            for ii in xrange(boxes.shape[0]):
                assert (boxes[ii, 2] < widths[i]), '[merge] fail at '+str(ii)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        assert False, 'not implemented'
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'imagenet_' + self._year))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
 
        #debug
        #iimage=cv2.imread(self.image_path_from_index(index))
        #width = image.shape[1]
        image = Image.open(self.image_path_from_index(index))
        im_width = image.size[0]
        im_height = image.size[1]

        #print '_load_imagenet_annotation()'
        all_boxes = np.zeros((0, 4), dtype=np.uint16)
        all_gt_classes = np.zeros((0), dtype=np.int32)
        all_overlaps = np.zeros((0, self.num_classes), dtype=np.float32)
        for class_id in self._class_ids:
            if class_id == '__background__':
                continue
            #class_id = re.sub(r'_[0-9]+','',index)
            #image_num = re.sub(r'n[0-9]+_','',index)
            filename = os.path.join(self._data_path, 'Annotations', class_id, index + '.xml')
            if not  os.path.exists(filename):
                continue
            # print 'Loading: {}'.format(filename)
            def get_data_from_tag(node, tag):
                return node.getElementsByTagName(tag)[0].childNodes[0].data
        
            with open(filename) as f:
                data = minidom.parseString(f.read())

            width=-1
            height=-1
            size = data.getElementsByTagName('size')
            num_size = len(size)
            assert num_size==1
            for ix, obj in enumerate(size):
                width = int(get_data_from_tag(obj, 'width'))
                height = int(get_data_from_tag(obj, 'height'))
            if width!=im_width or height!=im_height:
                print 'mismatch image: '+index
                return {'boxes' : np.zeros((0, 4), dtype=np.uint16),
                    'gt_classes': np.zeros((0), dtype=np.int32),
                    'gt_overlaps' : np.zeros((0, self.num_classes), dtype=np.float32),
                    'flipped' : False}

            objs = data.getElementsByTagName('object')
            num_objs = len(objs)
            if num_objs>0:
		  
		    cls_check = self._class_id_to_ind[class_id]
		    # Load object bounding boxes into a data frame.
		    for ix, obj in enumerate(objs):
			cls_id=str(get_data_from_tag(obj, "name")).lower().strip()
			if cls_id not in self._class_ids:
                            continue
                        #print ix
			#print obj
			# Make pixel indexes 0-based
			x1 = float(get_data_from_tag(obj, 'xmin')) #- 1
			y1 = float(get_data_from_tag(obj, 'ymin')) #- 1
			x2 = float(get_data_from_tag(obj, 'xmax')) #- 1
			y2 = float(get_data_from_tag(obj, 'ymax')) #- 1
                        if x2>=width or y2>=height:
                            print 'Bad bb on '+filename
                            continue
			assert not (x1==0 and y1==0 and x2==0 and y2==0), 'blank bb for '+index
			assert x1<=x2 and y1<=y2, 'bad bb for '+index
                        assert x1>=0 and x2>=0
			if not cls_id==class_id:
			    continue
			cls = self._class_id_to_ind[cls_id]
			assert not cls==0, 'bg class for '+index
			#assert cls==cls_check, 'Erronous assumption, bb for other classes present('+str(cls)+'!='+str(cls_check)+'):' + filename
			#cls = self.classes[self._class_id_to_ind[cls]]
			#print 'cls = ' + str(cls)
		        boxes = np.zeros((1, 4), dtype=np.uint16)
		        gt_classes = np.zeros((1), dtype=np.int32)
		        overlaps = np.zeros((1, self.num_classes), dtype=np.float32)
			boxes[0, :] = [x1, y1, x2, y2]
			gt_classes[0] = cls
			overlaps[0, cls] = 1.0
			
		        all_boxes = np.concatenate((all_boxes,boxes))
		        all_gt_classes = np.concatenate((all_gt_classes,gt_classes))
		        all_overlaps = np.concatenate((all_overlaps,overlaps))

        #assert not all_boxes.shape[0]==0, 'no bb found for '+index
        for bb in all_boxes:
           assert bb[0]<=bb[2] and bb[1]<=bb[3], 'bad bb for '+index
        for cls in all_gt_classes:
           assert cls>0 and cls<5, 'bg cls is ' +str(cls)+ ' for '+index
        
        #same-working debug Brian
        #for ii in range(0,all_overlaps.shape[0]):
        #    s=''
        #    for jj in range(0,all_overlaps.shape[1]):
        #        s+=str(all_overlaps[ii,jj])+', '
        #    print s
        #assert False
                
        boxes = all_boxes
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        #image=cv2.imread(self.image_path_at(i))
        #width = image.shape[1]
        #height = image.shape[0]
        for ii in xrange(boxes.shape[0]):
            assert (boxes[ii, 2] < width), '[load Ann] fail at '+str(ii)

        all_overlaps = scipy.sparse.csr_matrix(all_overlaps)
        return {'boxes' : all_boxes,
                'gt_classes': all_gt_classes,
                'gt_overlaps' : all_overlaps,
                'flipped' : False}

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'imagenet' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} imagenet results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'w') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id
    
    def _write_imagenet_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'imagenet' + self._year + '.txt')
        f=open(path, 'wt')
        for im_ind, index in enumerate(self.image_index):
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                # imagenet expects: <image_id> <ILSVRC2014_DET_ID> <confidence> <xmin> <ymin> <xmax> <ymax>
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, cls, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
        f.close()
        return comp_id

    def overlap(self,bb,bbgt):
        bi=(max(bb[0],bbgt[0]) , max(bb[1],bbgt[1]) , min(bb[2],bbgt[2]) , min(bb[3],bbgt[3]))
        iw=bi[2]-bi[0]+1.0
        ih=bi[3]-bi[1]+1.0
        ua=(bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih
        return iw*ih/ua
    def generate_PR_curves_imagenet_results(self, all_boxes):
        gt_roidb = self.gt_roidb()
        scores_and_is_tp = []
        total_tp=[]
        total_fp=[]
        for cls_ind, cls in enumerate(self.classes):
            scores_and_is_tp.append([])
            total_tp.append(0)
            total_fp.append(0)
        defaultIOUthr = 0.5
        pixelTolerance = 10
        for im_ind, index in enumerate(self.image_index):
            gt_classes = gt_roidb[im_ind]['gt_classes']
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                gt_cls_inds = [i for i in xrange(gt_classes.shape[0]) if gt_classes[i]==cls_ind]
                if len(gt_cls_inds)==0:
                    continue
                total_tp[cls_ind]+=len(gt_cls_inds)
                gt_cls_boxes = gt_roidb[im_ind]['boxes'][gt_cls_inds,:]
                
                gt_w=gt_cls_boxes[:,2]-gt_cls_boxes[:,0]
                gt_h=gt_cls_boxes[:,3]-gt_cls_boxes[:,1]
                thr = (gt_w*gt_h)/((gt_w+pixelTolerance)*(gt_h+pixelTolerance))
                for k in xrange(dets.shape[0]):
                    #is this detection true or false?
                    tp=0
                    ovmax=-1
                    jmax=-1
                    gt_detected = [0]*(gt_cls_boxes.shape[0])
                    for j in xrange(gt_cls_boxes.shape[0]):
                        if gt_detected[j]>0:
                            continue
                        thresh = min(thr[j],defaultIOUthr)
                        ov = self.overlap(dets[k,:],gt_cls_boxes[j,:])
                        if thresh<ov and ov>ovmax:
                            ovmax=ov
                            jmax=j
                            tp=1
                    if jmax>-1:
                        gt_detected[jmax]=1
                    if dets[k, -1]>0.9 and tp==0:
                        print 'fp on im '+index+' for class '+cls+' and box ['+str(dets[k, 0])+', '+str(dets[k, 1])+', '+str(dets[k, 2])+', '+str(dets[k, 3])+']'
                    scores_and_is_tp[cls_ind].append((dets[k, -1],tp))
        #
        def getScore(item):
            return item[0]
        ap=[0.0]*(len(self._class_ids)-1)
        for cls_ind, cls in enumerate(self._class_ids):
            if cls == '__background__':
                    continue
            is_tp = sorted(scores_and_is_tp[cls_ind], reverse=True, key=getScore)
            
            abovePos=0
            precision=[0.0]*len(is_tp)
            recall=[0.0]*len(is_tp)
            for ind, (score,tp) in enumerate(is_tp):
                #print 'score: '+str(score)+',  tp: '+str(tp)
                if tp==1:
                    abovePos+=1
                precision[ind]=abovePos/(ind+1.0)
                #if tp==1:
                ap[cls_ind-1]+=precision[ind]
                recall[ind] = abovePos/(total_tp[cls_ind]+0.0)
            ap[cls_ind-1]/=len(is_tp)
            print 'ap for class:'+cls+' is '+str(ap[cls_ind-1])
            pl.plot(recall,precision)
            pl.axis([0, 1, 0, 1])
            pl.ylabel('Precision')
            pl.xlabel('Recall')
            pl.show()
        mAP = 0.0
        for x in ap:
            mAP+=x
        mAP/=len(ap)
        print 'mAP is '+str(mAP)

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'imagenetdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self.generate_PR_curves_imagenet_results(all_boxes)
        #comp_id = self._write_voc_results_file(all_boxes)
        comp_id_2 = self._write_imagenet_results_file(all_boxes)
        #self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.my_imagenet('trainval', '2014')
    res = d.roidb
    from IPython import embed; embed()
