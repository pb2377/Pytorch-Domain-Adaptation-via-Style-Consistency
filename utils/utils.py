import os
import pickle
from torchvision import transforms
from PIL import Image, ImageDraw
from visdom import Visdom
import numpy as np
import torch


labelmap = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


def export_images(images, output_path, annos=None):
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
    if annos is not None:
        for idx in range(len(images)):
            image = plot_bounding_boxes(annos, images[idx], colour=(0, 255, 0))
            images[idx] = image
    image_out = Image.new(mode='RGB', size=(width, height))
    w = 0
    for idx, image in enumerate(images):
        h = (height - image.size[1]) // 2
        image_out.paste(image, (w, h))
        w += image.size[0]
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    image_out.save(output_path)


def plot_bounding_boxes(annos, image, colour):
    draw = ImageDraw.Draw(image)
    for anno in annos:
        draw.rectangle(anno[:-1], outline=colour, width=2)
        draw.text(anno[:2], str(int(anno[-1])) + ' ' + labelmap[int(anno[-1])], colour)
    return image


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')


def report_and_save(model, best_model, best_map, accuracy_history, output_dir, pseudolabel=False):
    # Average mAP over test points
    avg_map = []
    for acc_dict in accuracy_history:
        avg_map.append(acc_dict['mAP'])
    final_map = avg_map[-1]
    avg_map = np.mean(avg_map)
    std_map = np.std(avg_map)
    print('\nAveraged mAP over final 1000 iterations')
    print('AP = {:.4f} +/- {:.4f}'.format(avg_map, std_map))
    print('\nFinal mAP after {} iterations'.format(args.max_its))
    print('AP = {:.4f}'.format(final_map))
    print('\nBest mAP after final 1000 iterations')
    print('AP = {:.4f}'.format(best_map))

    # Save All Outputs
    # save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'weights',
                                                'ssd300-{}final.pth'.format('ps-' if pseudolabel else '')))

    # save best model
    torch.save(best_model, os.path.join(output_dir, 'weights',
                                        'ssd300-{}best.pth'.format('ps-' if pseudolabel else '')))

    # save accuracy history
    output_file = os.path.join(output_dir, 'accuracy_hist.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(accuracy_history, f)
