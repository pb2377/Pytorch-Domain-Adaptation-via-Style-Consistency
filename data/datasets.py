"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from random import shuffle
from numpy import random
from skimage.segmentation import slic
from skimage.color import label2rgb
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
VOC_ROOT = "../Datasets/VOC/data/VOCdevkit/"
assert osp.exists(VOC_ROOT)


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


class StylizedVOCDetection(VOCDetection):
    def __init__(self, root, target_domain,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712', stylized_root=None, mode='fast',
                 reduce_classes=False):
        super(StylizedVOCDetection, self).__init__(root, image_sets, transform, target_transform, dataset_name)
        assert stylized_root is not None
        self.mode = mode
        self.sigma = 2
        self.target_domain = target_domain

        self.reduce_classes = reduce_classes
        if target_domain in ['watercolor', 'comic'] and reduce_classes:
            self.remove_classes()

        # fast style transfer data path
        self._fastimgpath = osp.join(stylized_root, target_domain, 'tempstyle_%s', '%s', '%s.jpg')
        # cyclegan style transfer data path
        self._cyclepath = osp.join(stylized_root, 'dt_cycleGAN', 'dt_{}'.format(target_domain), '%s',
                                   'JPEGImages', '%s.jpg')

        if mode == 'cyclegan':
            self.cycle_ids = []
            for idx, img_id in enumerate(self.ids):
                idx_year = img_id[0].split('/')[-1]
                self.cycle_ids.append((osp.join(osp.dirname(stylized_root),
                                                'dt_cycleGAN', 'dt_{}'.format(target_domain), idx_year), img_id[-1]))
            self.pull_style = self.pull_cycle
        else:
            self.pull_style = self.pull_fast

        self.style_ids = self._get_style_ids()

        for idx in range(len(self.ids)):
            assert self.ids[idx][-1] == self.style_ids[idx][-1]

        print("{} Examples in Stylized VOC dataset".format(len(self.ids)))

    def __getitem__(self, index):
        im, style_im, gt, h, w = self.pull_item(index)
        return im, style_im, gt

    def _get_style_ids(self):
        style_ids = []
        for img_id in self.ids:
            assert len(img_id) == 2
            img_set = osp.basename(img_id[0])
            img_id = img_id[-1]
            if self.mode != 'cyclegan':
                style_ids.append((0, img_set, img_id))
            else:
                style_ids.append((img_set, img_id))
        return style_ids

    def _iterate_style_id(self, idx):
        self.style_ids[idx] = (self.style_ids[idx][0] + 1, self.style_ids[idx][1], self.style_ids[idx][2])
        if not osp.exists(self._fastimgpath % self.style_ids[idx]):
            self.style_ids[idx] = (0, self.style_ids[idx][1], self.style_ids[idx][2])

    def filter_target(self, target):
        # filter out redundant classes if using reduced classes on appropriate styles.
        updated_tar = []
        for bbox in target:
            if bbox[-1] in self.valid_indices:
                updated_tar.append(bbox)
        return updated_tar

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = self.pull_photo(index)
        height, width, channels = img.shape
        style_img = self.pull_style(index)

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
            if self.reduce_classes:
                target = self.filter_target(target)

        if style_img is None:
            if torch.cuda.is_available():
                raise NotImplementedError('Style Image Not Found')
            style_img = img
            # raise ValueError('Style Image Not Found.')

        if style_img.shape != (height, width, 3):
            style_img = fix_size(style_img, height, width)

        if self.transform is not None:
            target = np.array(target)
            img, style_img, boxes, labels = self.transform(img, style_img, target[:, :4], target[:, 4])

            img = img[:, :, (2, 1, 0)].astype(np.float32)
            style_img = style_img[:, :, (2, 1, 0)].astype(np.float32)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        img = torch.from_numpy(img).permute(2, 0, 1)
        style_img = torch.from_numpy(style_img).permute(2, 0, 1)
        return img, style_img, target, height, width

    def pull_image_and_info(self, index):
        '''Returns the original image object at index in PIL form and img_id

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        img = img[:, :, (2, 1, 0)]
        return torch.from_numpy(img).permute(2, 0, 1).float(), img_id

    def pull_photo(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        return img

    def pull_fast(self, index):
        style_file = self._fastimgpath % self.style_ids[index]
        img = cv2.imread(style_file)
        self._iterate_style_id(index)
        return img

    def pull_cycle(self, index):
        img_id = self.cycle_ids[index]
        img = cv2.imread(self._imgpath % img_id)
        return img

    def remove_classes(self):
        self.reduce_classes = True
        class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        data_classes = ['bicycle', 'bird', 'car', 'cat', 'dog', 'person']
        self.valid_indices = [class_to_ind[class_name] for class_name in data_classes]
        idx = 0
        while idx < len(self.ids):
            img_id = self.ids[idx]
            target = ET.parse(self._annopath % img_id).getroot()
            target = self.target_transform(target, 500, 500)
            iterate_idx = True
            for bbox in target:
                if bbox[-1] not in self.valid_indices:
                    self.ids.pop(idx)
                    iterate_idx = False
                    break
            if iterate_idx:
                idx += 1


def fix_size(image, h, w):
    delta_h = int(round(abs(image.shape[0] - h) / 2))
    delta_w = int(round(abs(image.shape[1] - w) // 2))
    image = image[delta_h:delta_h+h, delta_w:delta_w+w]
    return image


class ArtDetection(VOCDetection):
    """Clipart-watercolor-comic Detection Dataset Object

    input is image, target is annotation

    Arguments:

    """

    def __init__(self, root, target_domain, set_type=None, transform=None, target_transform=VOCAnnotationTransform()):
        super(ArtDetection, self).__init__(VOC_ROOT, [('2007', 'trainval'), ('2012', 'trainval')], None,
                                           VOCAnnotationTransform(), 'VOC0712')
        assert set_type in ['train', 'test', 'all']
        self.root = root
        self.transform = transform
        self.target_domain = target_domain
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        rootpath = osp.join(self.root, target_domain)
        txt_file = set_type + '.txt'
        self._imgsetpath = osp.join(rootpath, 'ImageSets', 'Main', txt_file)
        for line in open(osp.join(rootpath, 'ImageSets', 'Main', txt_file)):
            self.ids.append((rootpath, line.strip()))

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), img_id, target, height, width


class PseudolabelDataset(data.Dataset):
    """
    Pseudolabelled examples from Clipart-watercolor-comic Detection Dataset Object
    """
    def __init__(self, pseudolabels, root, target_domain, transform=None, stylized_root=None, mode='fast',
                 set_type='train'):
        super(PseudolabelDataset, self).__init__()
        if mode == 'cyclegan':
            raise NotImplementedError('Not implemented CycleGAN style transfer for PS labels')
        self.mode = mode
        self.root = root
        self.transform = transform
        self.pseudolabels = pseudolabels
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self._styleimgpath = osp.join(stylized_root, target_domain, 'ps_tempstyle_%s', '%s', '%s.jpg')

        self.ids = list()
        rootpath = osp.join(self.root, target_domain)

        for key in pseudolabels:
            self.ids.append((rootpath, key))

        self.class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.style_ids = self._get_style_ids()

    def _get_style_ids(self):
        style_ids = []
        for img_id in self.ids:
            assert len(img_id) == 2
            img_set = osp.basename(img_id[0])
            img_id = img_id[-1]
            style_ids.append((0, img_set, img_id))
        return style_ids

    def __getitem__(self, index):
        im, style_im, _, gt, h, w = self.pull_item(index)
        return im, style_im, None, gt

    def __len__(self):
        return len(self.ids)

    def _iterate_style_id(self, idx):
        if self.mode == 'fast':
            self.style_ids[idx] = (self.style_ids[idx][0] + 1, self.style_ids[idx][1], self.style_ids[idx][2])
            if not osp.exists(self._styleimgpath % self.style_ids[idx]):
                self.style_ids[idx] = (0, self.style_ids[idx][1], self.style_ids[idx][2])

    def pull_item(self, index):
        img_id = self.ids[index]
        target = self.pseudolabels[img_id[-1]]


        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        style_file = self._styleimgpath % self.style_ids[index]
        style_img = cv2.imread(style_file)
        self._iterate_style_id(index)

        if style_img is None:
            if torch.cuda.is_available():
                raise NotImplementedError()
            style_img = img
            # raise ValueError('Style Image Not Found.')

        # normalize target examples
        target = self.target_transform(target, width, height)

        style_img = fix_size(style_img, height, width)

        seg_mask = None
        if self.transform is not None:
            target = np.array(target)
            img, style_img, boxes, labels = self.transform(img, style_img, target[:, :4], target[:, 4])
            # convert from brg to rgb
            img = img[:, :, (2, 1, 0)].astype(np.float32)
            style_img = style_img[:, :, (2, 1, 0)].astype(np.float32)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        seg_mask = 0
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(style_img).permute(2, 0, 1), target, height, width

    def pull_image_and_info(self, index):
        '''Returns the original image object at index in PIL form and img_id

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        img = img[:, :, (2, 1, 0)]
        return torch.from_numpy(img).permute(2, 0, 1).float(), img_id

    @staticmethod
    def target_transform(target, width, height):
        """
        Equivalent to VOCDetectionTransform, but not from .xml file
        """
        res = []
        for item in target:
            bndbox = []
            for i, cur_pt in enumerate(item[:-1]):
                if not isinstance(cur_pt, float):
                    cur_pt = float(cur_pt)
                    # scale height or width
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                else:
                    raise NotImplementedError
                cur_pt = max(0., cur_pt)
                cur_pt = min(1., cur_pt)
                bndbox.append(cur_pt)
            label_idx = item[-1]
            bndbox.append(label_idx - 1)
            res.append(bndbox)  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res


class StyleSampler:
    """
    Style sampler for preprocessing stylized datasets with Clipart-watercolor-comic Detection Dataset Object
    """

    def __init__(self, root, target_domain, set_type):
        assert set_type in ['train', 'test', 'all']
        self.root = root
        self.target_domain = target_domain
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        rootpath = osp.join(self.root, target_domain)
        txt_file = set_type + '.txt'
        for line in open(osp.join(rootpath, 'ImageSets', 'Main', txt_file)):
            self.ids.append((rootpath, line.strip()))
        self.idx = [i for i in range(len(self.ids))]
        shuffle(self.idx)

    def __call__(self):
        if len(self.idx) < 1:
            self.idx = [i for i in range(len(self.ids))]
            shuffle(self.idx)
        im = self.pull_item(self.idx[0])
        self.idx.pop(0)
        return im

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        # print(self._imgpath % img_id)
        img = cv2.imread(self._imgpath % img_id)
        img = img[:, :, (2, 1, 0)]
        return torch.from_numpy(img).permute(2, 0, 1).float()
