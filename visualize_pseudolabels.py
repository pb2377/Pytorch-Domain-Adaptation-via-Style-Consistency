import json
import os
import os.path as osp
import sys

from PIL import Image, ImageDraw

from data import VOCAnnotationTransform

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def main():
    target_domain = 'watercolor'
    set_type = 'train'
    json_path = 'outputs/VGG-voc-watercolor-styles-pseudolabel-nopreservecolour/pslabels.json'
    root = '../Datasets/Clipart-Watercolor-Comic/{}'.format(target_domain)
    _imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
    _annopath = osp.join('%s', 'Annotations', '%s.xml')

    with open(json_path) as json_file:
        data = json.load(json_file)

    output_path = osp.join(osp.dirname(json_path), 'ps_images')
    if not osp.exists(output_path):
        os.makedirs(output_path)

    target_transform = VOCAnnotationTransform()
    for img_id in data:
        image = Image.open(_imgpath % (root, img_id))
        width, height = image.size

        # psuedolabel
        ps_anno = data[img_id]

        # groundtruth anno
        gt_annos = ET.parse(_annopath % (root, img_id)).getroot()
        gt_annos = target_transform(gt_annos, width, height)

        int_annos = []
        for gt_anno in gt_annos:
            new_anno = [gt_anno[0] * width, gt_anno[1] * height,
                        gt_anno[2] * width, gt_anno[3] * height, gt_anno[4]]
            new_anno = [int(round(i)) for i in new_anno]
            int_annos.append(new_anno)

        gt_anno = int_annos

        ps_image = draw_annos(image.copy(), ps_anno, colour=(0, 0, 255))
        gt_image = draw_annos(image.copy(), gt_anno, colour=(0, 255, 0))

        export_image([ps_image, gt_image], output_path=osp.join(output_path, img_id + '.png'))
        # raise NotImplementedError

    pass


def export_image(images, output_path, annos=None):
    if not isinstance(images, list):
        images = [images]
    height = 0
    width = 0
    for image in images:
        height = max(image.size[1], height)
        width += image.size[0]
    # normalize to [0, 1]
    # for idx, image in enumerate(images):
    #     images[idx] = images[idx] / 255.

    # transf = transforms.ToPILImage()
    # images = [transf(image.squeeze().cpu()) for image in images]

    # if annos is not None:
    #     for idx, image in enumerate(images):
    #         images[idx] = draw_annos(image, annos)

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
        # anno = anno.tolist()
        if max(anno[:-1]) <= 1:
            anno[0] = int(anno[0] * w)
            anno[1] = int(anno[1] * h)
            anno[2] = int(anno[2] * w)
            anno[3] = int(anno[3] * h)
        anno = [int(i) for i in anno]
        drw.rectangle(anno[:-1], outline=colour)
        drw.text(anno[:2], VOC_CLASSES[anno[-1]], colour=colour)
    return image


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

if __name__ == '__main__':
    main()
