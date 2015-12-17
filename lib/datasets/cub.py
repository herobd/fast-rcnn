# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Brian Davis
# --------------------------------------------------------

import datasets
import datasets.cub
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


class cub(datasets.imdb):
    def __init__(self, image_set, year, devkit_path=None):
        datasets.imdb.__init__(self, 'CUB_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path)
	self._readClassesAndIndex()
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_name_to_ind = dict(zip(self._class_names, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()

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
                'CUBdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
        self.createCaffeFile()

    def _readClassesAndIndex(self):
        f=open(os.path.join(self._data_path,'classes.txt'),'r')
        classes = ['__background__']
        class_names=['__background__']
        for line in f:
            m = re.search(r'(\d+)\s+\d\d\d\.(\w+)',line);
            classes.append(m.group(1))
            class_names.append(m.group(2))
        self._classes = tuple(classes)
        self._class_names =tuple(class_names)
        f.close()

        f=open(os.path.join(self._data_path,'images.txt'),'r')
        self._image_path_lookup=[]
        for line in f:
            m = re.search(r'(\d+)\s(.+)\s',line);
            assert int(m.group(1))-1==len(self._image_path_lookup)
            self._image_path_lookup.append(m.group(2))
        f.close()
    
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
        image_path = os.path.join(self._data_path, 'images',
                                  self._image_path_lookup[index])
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
        image_set_file = os.path.join(self._data_path, 'train_test_split.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        image_index=[]
        with open(image_set_file) as f:
            for line in f:
                m=re.search(r'(\d+)\s+([01])',line)
                if (self._image_set=='train' and m.group(2)=='1') or (self._image_set=='test' and m.group(2)=='0'):
                    image_index.append(int(m.group(1))-1)
                    
        return image_index
    

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'CUB_200_' + self._year)

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

        #gt_roidb = [self._load_imagenet_annotation(index)
        #            for index in self.image_index]
        gt_roidb = self._load_CUB_annotations()
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
                assert cls>0 and cls<self.num_classes, 'cls is ' +str(cls)+ ' at index: '+str(ii)
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

        if int(self._year) == 2011 or self._image_set != 'test':
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

    def createCaffeFile(self):
        image_set_file = os.path.join('/home/brianld/fast-rcnn/data/CUB_200_2011',
                                      'all_'+self._image_set + '.txt')
        gt_roidb=self.gt_roidb()
        f=open(image_set_file,'w')
        for (im_index,i) in enumerate(self._image_index):
            path = self._image_path_lookup[i]
            assert gt_roidb[im_index]['gt_classes'].shape[0]==1
            cls = self._classes[gt_roidb[im_index]['gt_classes'][0]]
            f.write(path+' '+cls+'\n')
        f.close()

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        if not os.path.exists(filename):
            image_set_file = os.path.join('/home/brianld/fast-rcnn/lib/datasets/', 'tmp',
                                      self._image_set + '.txt')
            f=open(image_set_file,'w')
            for i in self._image_index:
                path = self._image_path_lookup[i]
                f.write(path+'\n')
            f.close()
            assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
            image_loc = os.path.join('/home/brianld/fast-rcnn','data', 'CUB_200_' + self._year, 'images')
            
            cmd = 'cd {} && '.format('/home/brianld/fast-rcnn/lib/datasets/')
            cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
            cmd += '-r "dbstop if error; '
            cmd += 'do_selective_search(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
                   .format(image_set_file, image_loc,'', filename)
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

    def _load_CUB_annotations(self):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        ret=[]
        f_labels = open(os.path.join(self._data_path,'image_class_labels.txt'))
        f_bbs = open(os.path.join(self._data_path,'bounding_boxes.txt'))
        labels=f_labels.readlines()
        bbs=f_bbs.readlines()
        assert(len(labels)==len(bbs))
        
        sizes = [Image.open(self.image_path_at(i)).size
                  for i in xrange(self.num_images)]

	for (index,i) in enumerate(self._image_index):
            m_label = re.search(r'(\d+)\s+(\d+)',labels[i])
            cls = int(m_label.group(2))
            img = int(m_label.group(1))-1
            m_bb = re.search(r'(\d+)\s+(\d*\.?\d+)\s+(\d*\.?\d+)\s+(\d*\.?\d+)\s+(\d*\.?\d+)',bbs[i])
            assert (img == int(m_bb.group(1))-1 and img==i)

            boxes = np.zeros((1, 4), dtype=np.uint16)
            x1=float(m_bb.group(2))
            y1=float(m_bb.group(3))
            x2=x1+float(m_bb.group(4))-1
            y2=y1+float(m_bb.group(5))-1
            if x2>=sizes[index][0]:
                assert x2-sizes[index][0]<40
                x2=sizes[index][0]-1
            if y2>=sizes[index][1]:
                assert y2-sizes[index][1]<40
                y2=sizes[index][1]-1
            boxes[0, :] = [x1, y1, x2, y2]
            
            gt_classes = np.zeros((1), dtype=np.int32)
            gt_classes[0] = cls

            overlaps = np.zeros((1, self.num_classes), dtype=np.float32)
            overlaps[0, cls] = 1.0
            overlaps = scipy.sparse.csr_matrix(overlaps)
            
            ret.append( {'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps' : overlaps,
                    'flipped' : False} )
        return ret

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
        path = os.path.join(self._devkit_path, 'results', 'CUB' + self._year + '.txt')
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
    def score_results(self, all_boxes):
        confusion = np.zeros([self.num_classes,self.num_classes])
        gt_roidb = self.gt_roidb()
        correct=0
        for im_ind, index in enumerate(self.image_index):
            
            gt_classes = gt_roidb[im_ind]['gt_classes']
            assert(gt_classes.shape[0]==1);
            gt_class = gt_classes[0]
            highestScore=0
            highestClass=-1
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                maxVal = np.amax(dets,0)
                score = maxVal[4]
                if score > highestScore:
                    highestScore = score
                    highestClass = cls_ind
            confusion[gt_class,highestClass]+=1
            if gt_class == highestClass:
                correct+=1
        print 'accuracy = ' + str(correct/float(len(self.image_index)))
        maxC=0
        minC=len(self.image_index)*9
        for r in xrange(self.num_classes):
            for c in xrange(self.num_classes):
                if (confusion[r,c]<minC):
                    minC=confusion[r,c]
                if (confusion[r,c]>maxC):
                    maxC=confusion[r,c]
        for r in xrange(self.num_classes):
            for c in xrange(self.num_classes):
                confusion[r,c] = 255*(confusion[r,c]-minC)/float(maxC)
        return confusion
        #

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
        path = os.path.join(self._devkit_path, 'results', 'all_boxes_CUB' + self._year + '.pkl')
        fid=open(path,'wb')
        cPickle.dump(all_boxes, fid, cPickle.HIGHEST_PROTOCOL)
        confusion = self.score_results(all_boxes)
        #comp_id = self._write_voc_results_file(all_boxes)
        comp_id_2 = self._write_imagenet_results_file(all_boxes)
        #self._do_matlab_eval(comp_id, output_dir)
        cv2.imwrite(os.path.join(output_dir,'confusion_'+self._image_set+'.png'),confusion)

    def evaluate_detections_from(self, path):
        assert os.path.exists(path)
        fid=open(path,'rb')
        all_boxes=cPickle.load(fid)
        confusion = self.score_results(all_boxes)
        cv2.imwrite(os.path.join('output','confusion_'+self._image_set+'.png'),confusion)
    
    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.cub('trainval', '2011')
    res = d.roidb
    from IPython import embed; embed()
