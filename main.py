import argparse
import os
import pickle

import torch

import eval
import models
import train_pseudolabel
import trainer
from cfg import cfg
from data import datasets, BaseTransform, VOCAnnotationTransform

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
# add args
parser.add_argument('--train', default=False, action='store_true', help='Flag to train model.')
parser.add_argument('--target_domain', default='clipart', choices=['clipart', 'watercolor', 'watercolour', 'comic'],
                    type=str, help='Target domain to be adapted to.')
parser.add_argument('--batch_size', default=6, type=int,
                    help='Batch size for training')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--overlap_thresh', default=0.45, type=float,
                    help='Overlap threshold for detection in test and pseudolabelling.')
parser.add_argument('--pthresh', default=0.7, type=float,
                    help='Positive threshold confidence for pseudolabelling.')
parser.add_argument('--nthresh', default=0.9, type=float,
                    help='Negative threshold confidence for mining during pseudolabel training.')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--stage1_its', default=10000, type=int, help='Max training iterations in stage 1 ')
parser.add_argument('--stage2_its', default=5000, type=int, help='Max training iterations in stage 2')
parser.add_argument('--test_freq', default=100, type=int, help='Frequency to test model at the end of training.')
parser.add_argument('--log_freq', default=100, type=int, help='Frequency to report training losses.')
parser.add_argument('--style_root', default='../Datasets/Clipart-Watercolor-Comic/',
                    help='Root for clipart-watercolor-comic dataset.')
parser.add_argument('--voc_root', default='../Datasets/VOC/data/VOCdevkit/', help='Root for VOC dataset.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--output_dir', default=None,
                    help='Directory for outputting validation results, etc')
parser.add_argument('--mode', default='fast', type=str,
                    choices=['fast', 'cyclegan', 'mixed'],
                    help='Which style transfer method to use.')
parser.add_argument('--preserve_colours', default='nopreserve', type=str,
                    choices=['nopreserve', 'preserve', 'random'],
                    help='When style transferring with fast method, preserve source colour or not, '
                         'or randomly choose per example')
parser.add_argument('--no_prep', default=False, action='store_true', help='Skip style transfer preprocessing.')
args = parser.parse_args()


def main(args, cfg):
    print(args)

    if args.mode == 'watercolour':
        args.mode = 'watercolor'

    num_classes = 21

    phase = 'train' if args.train else 'test'
    model = models.vgg_ssd.build_ssd(cfg, phase=phase, size=300, num_classes=num_classes)

    # # Load pretrained weights
    trained_model = "weights/ssd300_mAP_77.43_v2.pth"
    model_dict = model.state_dict()
    model_dict.update(torch.load(trained_model, map_location='cpu'))
    model.load_state_dict(model_dict)

    # setup output dir for train/test logs
    if args.output_dir is None:
        args.output_dir = 'outputs-{}'.format(args.target_domain)
    print("\nSaving outputs to {}".format(args.output_dir))

    # Find root directory for style transferred examples
    stylized_root = '../Datasets/DA_transfer'
    if args.mode != 'cyclegan':
        if args.preserve_colours == 'random':
            stylized_root = os.path.join(stylized_root, 'fast_random_colour')
        elif args.preserve_colours == 'preserve':
            stylized_root = os.path.join(stylized_root, 'fast_preserve_colour')
        else:
            stylized_root = os.path.join(stylized_root, 'fast_no_preserve_colour')

    if args.train:
        # Base training with VOC images and stylized versions, then pseudolabel and continue training.

        print("\nTraining Base Model on Stylized Photos pairs....")
        # if not args.pseudolabel:
        model = trainer.base_trainer(model, args, args.output_dir, stylized_root, num_classes)

        print("\nTraining with Joint Dataset of Pseudolabelled Art and Stylized Photos pairs....")
        model = train_pseudolabel.pseudolabel_trainer(model, args, args.output_dir, stylized_root, num_classes)
    else:
        assert args.checkpoint is not None
        # load weights
        assert os.path.exists(args.checkpoint)
        state_dict_to_load = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict_to_load)
        # load base model weights
        print('Model loaded.')

    # create test dataset
    dataset_mean = (104, 117, 123)
    set_type = 'test'
    eval_dataset = datasets.ArtDetection(root=args.style_root, transform=BaseTransform(300, dataset_mean),
                                         target_domain=args.target_domain, set_type=set_type,
                                         target_transform=VOCAnnotationTransform())

    # set up dataloaders
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    accuracy_dict = eval.evaluate(model, eval_dataset, os.path.join(args.output_dir, 'evaluate'))

    output_file = os.path.join(args.output_dir, 'final_accuracy_dict.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(accuracy_dict, f)


if __name__ == '__main__':
    main(args, cfg)
