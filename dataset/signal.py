import os
import numpy as np
from imdb import Imdb
from evaluate.signal_eval import signal_eval
import cv2
import cPickle


class Signal(Imdb):
    """
    Base class for loading datasets as used in YOLO

    Parameters:
    ----------
    name : str
        name for this dataset
    classes : list or tuple of str
        class names in this dataset
    list_file : str
        filename of the image list file
    image_dir : str
        image directory
    label_dir : str
        label directory
    extension : str
        by default .jpg
    label_extension : str
        by default .txt
    shuffle : bool
        whether to shuffle the initial order when loading this dataset,
        default is True
    """
    def __init__(self, name, classes, list_file, base_dir, image_dir, label_dir, \
                 extension='.jpg', label_extension='.txt', shuffle=True):
        if isinstance(classes, list) or isinstance(classes, tuple):
            num_classes = len(classes)
        elif isinstance(classes, str):
            with open(classes, 'r') as f:
                classes = [l.strip() for l in f.readlines()]
                num_classes = len(classes)
        else:
            raise ValueError, "classes should be list/tuple or text file"
        assert num_classes > 0, "number of classes must > 0"
        super(Signal, self).__init__(name + '_' + str(num_classes))
        self.classes = classes
        self.num_classes = num_classes
        self.list_file = os.path.join(base_dir, list_file)
        self.image_dir = os.path.join(base_dir, image_dir)
        self.label_dir = os.path.join(base_dir, label_dir)
        self.cache_path = os.path.join(base_dir, 'cache')
        self.base_dir = base_dir
        self.extension = extension
        self.label_extension = label_extension

        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        self.labels = self._load_image_labels()


    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        assert os.path.exists(self.list_file), 'Path does not exists: {}'.format(self.list_file)
        with open(self.list_file, 'r') as f:
            image_set_index = [x.strip().split('/')[-1].split('.')[0] for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.image_dir, name) + self.extension
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index, :, :]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.label_dir, index + self.label_extension)
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []
        max_objects = 0

        # load ground-truths
        for idx in self.image_set_index:
            label_file = self._label_path_from_index(idx)
            with open(label_file, 'r') as f:
                label = []
                for line in f.readlines():
                    temp_label = line.strip().split()
                    assert len(temp_label) == 5, "Invalid label file" + label_file
                    cls_id = int(temp_label[0])
                    x = float(temp_label[1])
                    y = float(temp_label[2])
                    half_width = float(temp_label[3]) / 2
                    half_height = float(temp_label[4]) / 2
                    xmin = x - half_width
                    ymin = y - half_height
                    xmax = x + half_width
                    ymax = y + half_height
                    #another version convert of dataset
                    #cls_id, xmin, xmax, ymin, ymax = map(float, temp_label)
                    #cls_id = int(cls_id)
                    #end version convert
                    label.append([cls_id, xmin, ymin, xmax, ymax])
                temp.append(np.array(label))
                max_objects = max(max_objects, len(label))
        # add padding to labels so that the dimensions match in each batch
        assert max_objects > 0, "No objects found for any of the images"
        self.padding = max_objects
        labels = []
        for label in temp:
            if len(label) != 0:
                label = np.lib.pad(label, ((0, max_objects-label.shape[0]), (0,0)), \
                               'constant', constant_values=(-1, -1))
            labels.append(label)
        return np.array(labels)
    
    def evaluate_detections(self, detections):
        """
        top level evaluations
        Parameters:
        ----------
        detections: list
            result list, each entry is a matrix of detections
        Returns:
        ----------
            None
        """
        # make all these folders for results
        result_dir = os.path.join(self.base_dir, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        #year_folder = os.path.join(self.base_dir, 'results', 'VOC)
        #if not os.path.exists(year_folder):
        #    os.mkdir(year_folder)
        #res_file_folder = os.path.join(self.base_dir, 'results', 'Main')
        #if not os.path.exists(res_file_folder):
        #    os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        self.do_python_eval()
        

    def get_result_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = 'signal_det_valid_{:s}.txt'
        dirpath = os.path.join(self.base_dir,'results')
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        
        path = os.path.join(dirpath, filename)
        return path

    def write_pascal_results(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            print 'Writing {} VOC results file'.format(cls)
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[im_ind]
                    if dets.shape[0] < 1:
                        continue
                    h, w = self._get_imsize(self.image_path_from_index(im_ind))
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        if (int(dets[k, 0]) == cls_ind):
                            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                    format(index, dets[k, 1],
                                           int(dets[k, 2] * w) + 1, int(dets[k, 3] * h) + 1,
                                           int(dets[k, 4] * w) + 1, int(dets[k, 5] * h) + 1))

    def do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self.base_dir,
            'labels',
            '{:s}.txt')
        imagesetfile = os.path.join(
            self.base_dir,
            #self.list_file + '.txt')
            self.list_file)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        cachedir = os.path.join(self.cache_path, self.name)
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric =  False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self.classes):
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = signal_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.3,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        
    def _get_imsize(self, im_name):
        """
        get image size info
        Returns:
        ----------
        tuple of (height, width)
        """
        img = cv2.imread(im_name)
        return (img.shape[0], img.shape[1])
        
   
