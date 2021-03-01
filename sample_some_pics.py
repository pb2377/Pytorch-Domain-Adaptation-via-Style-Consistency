import os
import sys
import os.path as osp
from shutil import copyfile
from data import VOC_CLASSES, VOCAnnotationTransform
import random
import cv2
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def main():
    # voc ids
    voc_root = '../Datasets/VOC/data/VOCdevkit/'
    style_root = '../Datasets/Clipart-Watercolor-Comic/'
    image_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    list_ids = list()
    for (year, name) in image_sets:
        rootpath = osp.join(voc_root, 'VOC' + year)
        for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
            list_ids.append((rootpath, line.strip()))

    _annopath = osp.join('%s', 'Annotations', '%s.xml')
    _imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')

    random.shuffle(list_ids)
    base_dir = 'example_images/voc'
    sample_images(list_ids, base_dir, _annopath, _imgpath)

    for target_domain in ['clipart', 'watercolor', 'comic']:
        list_ids = list()
        rootpath = osp.join(style_root, target_domain)
        txt_file = 'train.txt'
        for line in open(osp.join(rootpath, 'ImageSets', 'Main', txt_file)):
            list_ids.append((rootpath, line.strip()))

        random.shuffle(list_ids)
        base_dir = osp.join('example_images', target_domain)
        sample_images(list_ids, base_dir, _annopath, _imgpath)


ind_to_class = dict(zip(range(len(VOC_CLASSES)), VOC_CLASSES))
class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
data_classes = ['bicycle', 'bird', 'car', 'cat', 'dog', 'person']
valid_indices = [class_to_ind[class_name] for class_name in data_classes]
anno_trans = VOCAnnotationTransform()


def sample_images(list_ids, base_dir, _annopath, _imgpath):
    for cls in valid_indices:
        # cls_dir = osp.join(base_dir, ind_to_class[cls])
        cls_dir = base_dir
        os.makedirs(cls_dir, exist_ok=True)
        idx = 0
        nims = 0
        while nims < 5 and idx < len(list_ids):
            img_id = list_ids[idx]
            image = cv2.imread(_imgpath % img_id)
            height, width, channels = image.shape
            annos = ET.parse(_annopath % img_id).getroot()
            annos = anno_trans(annos, height, width)
            anidx = 0
            while anidx < len(annos):
                if not annos[anidx][-1] == cls:
                    annos.pop(anidx)
                else:
                    anidx += 1

            if len(annos) > 0:
                anno = random.choice(annos)
                # crop instances and save
                x = int(anno[0] * height)
                xx = int(anno[2] * height)
                y = int(anno[1] * width)
                yy = int(anno[3] * width)
                if xx - x >= 100 and yy - y >= 100:
                    image_out = image[y:yy, x:xx]
                    output_path = osp.join(cls_dir, '{}_{}.jpg'.format(img_id[-1], cls))
                    cv2.imwrite(output_path, image_out)
                    nims += 1
            idx += 1


if __name__ == '__main__':
    main()