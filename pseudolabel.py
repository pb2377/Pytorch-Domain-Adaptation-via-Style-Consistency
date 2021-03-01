from __future__ import print_function
from data import *
import torch

from torch.autograd import Variable

from torchvision import transforms
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image, ImageDraw
import torch.optim as optim

import numpy as np


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


def pseudolabel(net, dataset, pthresh, overlap_thresh=0.45):
    """Pseudolabel positives"""
    net.eval()
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    pseudolabels = {}
    n_labels = 0
    for i in range(num_images):
        im, img_id, gt, h, w = dataset.pull_item(i)
        img_id = img_id[-1]
        x = Variable(im.unsqueeze(0))
        if torch.cuda.is_available():
            x = x.cuda()

        detections = net(x).data
        # skip j = 0, because it's the background class

        target = []
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()

            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            # all_boxes[j][i] = cls_dets
            # print(cls_dets)
            nms_boxes = []
            nms_scores = []
            for item in cls_dets:
                nms_boxes.append(item[:-1].tolist())
                nms_scores.append(item[-1:].tolist())
                # if item[-1] > pthresh:
                #     nms_boxes.append(item[:-1].tolist())
                #     nms_scores.append(item[-1:].tolist())

            # boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
            # scores: (tensor) The class predscores for the img, Shape:[num_priors].
            if len(nms_scores) > 0:
                # nms_boxes = np.array(nms_boxes)
                # nms_scores = np.array(nms_scores)[:, 0]
                # keep = nms(nms_boxes, nms_scores, thresh=overlap_thresh)
                # print(len(nms_scores), len(keep))
                # nms_boxes = nms_boxes[keep].tolist()
                # nms_scores = nms_scores[keep].tolist()

                for idx, score in enumerate(nms_scores):
                    # Only accept pseudolabels over the positive threshold
                    if score[0] > pthresh:
                        bbox = nms_boxes[idx]
                        bbox = [int(round(i)) for i in bbox]
                        bbox.append(j)
                        target.append(bbox)

            # n_labels += len(nms_scores)

        if len(target) > 0:
            pseudolabels[img_id] = target
        n_labels += len(target)

    print("--- {} Detections Pseudolabeled in {} Images ---".format(n_labels, len(pseudolabels.keys())))
    return pseudolabels


def export_image(images, output_path, annos=None):
    if not isinstance(images, list):
        images = [images]
    height = 0
    width = 0
    for image in images:
        height = max(image.size(-2), height)
        width += image.size(-1)
    # normalize to [0, 1]
    for idx, image in enumerate(images):
        images[idx] = images[idx] / 255.

    transf = transforms.ToPILImage()
    images = [transf(image.squeeze().cpu()) for image in images]

    if annos is not None:
        for idx, image in enumerate(images):
            images[idx] = draw_annos(image, annos)

    image_out = Image.new(mode='RGB', size=(width, height))
    w = 0
    for idx, image in enumerate(images):
        h = (height - image.size[1]) // 2
        image_out.paste(image, (w, h))
        w += image.size[0]
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    image_out.save(output_path)


def draw_annos(image, annos, colour=(0, 0, 255)):
    drw = ImageDraw.Draw(image)
    w, h = image.size
    for anno in annos:
        anno = anno.tolist()
        if max(anno[:-1]) <= 1:
            anno[0] = int(anno[0] * w)
            anno[1] = int(anno[1] * h)
            anno[2] = int(anno[2] * w)
            anno[3] = int(anno[3] * h)
        anno = [int(i) for i in anno]
        drw.rectangle(anno[:-1], outline=colour)
        drw.text(anno[:2], VOC_CLASSES[anno[-1]-1], colour=colour)
    return image


def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maximum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
