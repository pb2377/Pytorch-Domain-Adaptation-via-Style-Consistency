import os
import time
from random import shuffle

import torch
from PIL import Image
from torchvision import transforms

import styletransfer
from data.datasets import StyleSampler


def check_preprocess(args, dataloader, stylized_root, iterations, pseudolabel=False):
    if args.no_prep or args.mode != 'fast':
        pass
    else:
        preserve_colour = False
        if args.preserve_colours == 'random':
            preserve_colour = None
        elif args.preserve_colours == 'preserve':
            preserve_colour = True
        preprocess(dataloader.dataset, args.target_domain, iterations, args.batch_size,
                   args.style_root, stylized_root, set_type='train', preserve_colour=preserve_colour,
                   pseudo=pseudolabel)


def preprocess(base_dataset, target_domain, max_its, batch_size, style_root, stylized_root, set_type,
               preserve_colour=None, pseudo=False):
    # iterate until
    style_data = StyleSampler(root=style_root, target_domain=target_domain, set_type=set_type)
    style_transfer = styletransfer.StyleTransferModule(preserve_colour=preserve_colour)

    episode = 'ps_tempstyle' if pseudo else 'tempstyle'

    base_path = os.path.join(stylized_root, '{}/{}_'.format(target_domain, episode))

    # max_prep = ceil(max_its / (len(base_dataset) / float(batch_size)))
    max_prep = 10 if pseudo else 5
    shuffle_idx = [i for i in range(len(base_dataset))]

    t0 = time.time()
    print('Style transferring training examples {} times...'.format(max_prep))
    while len(shuffle_idx) > 0:
        shuffle(shuffle_idx)
        for idx in range(len(base_dataset)):
            if idx in shuffle_idx:
                if not idx % batch_size:
                    style_source = style_data()
                image, img_id = base_dataset.pull_image_and_info(idx)
                img_set = os.path.basename(img_id[0])
                img_id = img_id[1]
                outpath = None
                for save_id in range(max_prep):
                    temp_path = os.path.join(base_path + str(save_id), img_set, str(img_id) + '.jpg')
                    if not os.path.exists(temp_path):
                        if not os.path.exists(os.path.dirname(temp_path)):
                            os.makedirs(os.path.dirname(temp_path))
                        outpath = temp_path
                        break

                if outpath is not None:
                    with torch.no_grad():
                        style_im = style_transfer(image.unsqueeze(0), style_source.unsqueeze(0))
                    export_image(style_im, output_path=outpath)
                    del style_im, image
                else:
                    shuffle_idx.remove(idx)

    t1 = time.time()
    delta_t = (t1 - t0) / 3600
    print('     Finished Style Transfer on Dataset in {:.2f} Hours'.format(delta_t))
    del image, style_source, style_transfer


def export_image(images, output_path):
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
    #     im -= im.mean()
    #     im /= im.max()
    #     images[idx] = im

    transf = transforms.ToPILImage()
    images = [transf(image.squeeze().cpu()) for image in images]
    image_out = Image.new(mode='RGB', size=(width, height))
    w = 0
    for idx, image in enumerate(images):
        h = (height - image.size[1]) // 2
        image_out.paste(image, (w, h))
        w += image.size[0]
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    image_out.save(output_path)
