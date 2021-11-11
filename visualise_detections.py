"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function

import os
import pickle
import sys
import time

import numpy as np
import torch
from torchvision import transforms

from data import VOC_CLASSES as labelmap
from data import datasets

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

FULL_REPORT = True
MEANS = (104, 117, 123)


def main():
    target_domain = 'watercolor'
    det_file = 'outputs/watercolor-fast-nopreserve/iteration-9100/detections.pkl'.format(target_domain)
    style_root = '../Datasets/Clipart-Watercolor-Comic/'
    output_dir = os.path.dirname(det_file)
    val_data = datasets.ArtDetection(root=style_root, target_domain=target_domain, set_type='test',
                                     transform=None)

    test_net(det_file, output_dir, val_data, set_type='test')
    # with open(det_file, 'rb') as f:
    #     unpickler = pickle.Unpickler(f)
    #     all_boxes = unpickler.load()


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(dir_path):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(dir_path)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(outpath, image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(outpath, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset, outpath, set_type):
    for cls_ind, cls in enumerate(labelmap):
        if FULL_REPORT:
            print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(outpath, set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir, annopath, imgsetpath, target_domain, use_07=True, set_type=None, overlap_threshold=0.45):
    assert set_type is not None
    cachedir = os.path.join(output_dir, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    if FULL_REPORT:
        print('VOC07 metric? ' + ('Yes' if use_07 else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    accuracy_dict = {}
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(output_dir, set_type, cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imgsetpath.format(set_type), cls, cachedir,
            ovthresh=overlap_threshold, use_07_metric=use_07)
        aps += [ap]

        if FULL_REPORT:
            print('AP for {} = {:.4f}'.format(cls, ap))

        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        accuracy_dict[cls] = {}
        accuracy_dict[cls]['AP'] = ap
        accuracy_dict[cls]['Prec'] = prec
        accuracy_dict[cls]['Rec'] = rec

    # if target_domain in ['comic', 'watercolor']:
    #     indices = [1, 2, 6, 7, 11, 14]
    #     aps = [j for i, j in enumerate(aps) if i in indices]

    if FULL_REPORT:
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')
    else:
        print('Mean AP {:.4f} -- Outputs saved to {}'.format(np.mean(aps), output_dir))

    accuracy_dict['mAP'] = np.mean(aps)
    return accuracy_dict


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.45,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
       detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
       annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default True)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file§
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # if not os.path.isfile(cachefile):
    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath % (imagename))
        if i % 100 == 0:
            if FULL_REPORT:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
    # save
    if FULL_REPORT:
        print('Saving cached annotations to {:s}'.format(cachefile))

    with open(cachefile, 'wb') as f:
        pickle.dump(recs, f)
    # else:
    #     # load
    #     print('Loading from {}'.format(cachefile))
    #     with open(cachefile, 'rb') as f:
    #         recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(det_file, output_dir, dataset, set_type, use_07=True, conf_thresh=0.01):
    with open(det_file, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        all_boxes = unpickler.load()

    to_pil_img = transforms.ToPILImage()

    for idx in range(len(dataset)):
        # get image
        image, img_id, target, height, width = dataset.pull_item(idx)
        image = to_pil_img(image)
        image.show()
        for cls in range(len(all_boxes)):
            # get boxes
            raise NotImplementedError

    if FULL_REPORT:
        print('Evaluating detections')
    accuracy_dict = evaluate_detections(all_boxes, output_dir, dataset, set_type, use_07=use_07)
    return accuracy_dict


def evaluate_detections(box_list, output_dir, dataset, set_type, use_07=True):
    write_voc_results_file(box_list, dataset, output_dir, set_type)
    annopath = dataset._annopath % (os.path.join(dataset.root, dataset.target_domain), '%s')
    imgsetpath = dataset._imgsetpath
    accuracy_dict = do_python_eval(output_dir, annopath, imgsetpath, dataset.target_domain, use_07=use_07,
                                   set_type=set_type)
    return accuracy_dict


def evaluate(model, dataset, save_folder):
    accuracy_dict = test_net(save_folder, model, dataset, set_type='test', use_07=True, conf_thresh=0.01)
    return accuracy_dict


if __name__ == '__main__':
    main()
