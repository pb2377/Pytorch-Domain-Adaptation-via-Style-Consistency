import os
import argparse
import models
import torch
from data import datasets, style_detection_collate, BaseTransform, VOCAnnotationTransform
from models.layers.modules import MultiBoxLoss, FeatureConsistency
import torch.utils.data as data
import trainer
from cfg import cfg
from utils.augmentations import StyleAugmentation
import pickle
import json
import preprocess
import train_pseudolabel
import eval

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
# add args
parser.add_argument('--train', default=False, action='store_true', help='Flag to train model')
parser.add_argument('--pseudolabel', '--ps', default=False, action='store_true',
                    help='Flag to generate and train with pseudolabels from base model at --checkpoint.')
parser.add_argument('--eval', default=False, action='store_true', help='Flag to evaluate model')
parser.add_argument('--test', default=False, action='store_true', help='Flag to test model i.e. label and visualize '
                                                                       'target dataset')
parser.add_argument('--target_domain', default='clipart', choices=['clipart', 'watercolor', 'watercolour', 'comic'],
                    type=str, help='Target domain to be adapted to.')
parser.add_argument('--weights', required=False, default=None, type=str, help='path to initial weights')
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
parser.add_argument('--max_its', default=None, type=int, help='Max training iterations')
parser.add_argument('--test_freq', default=100, type=int, help='Frequency to test model at the end of training.')
parser.add_argument('--log_freq', default=100, type=int, help='Frequency to report training losses.')
parser.add_argument('--style_root', default='../Datasets/Clipart-Watercolor-Comic/',
                    help='Root for clipart-watercolor-comic dataset.')
parser.add_argument('--voc_root', default='../Datasets/VOC/data/VOCdevkit/', help='Root for VOC dataset.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--output_dir', default=None,
                    help='Directory for outputting validation results, etc')
parser.add_argument('--photometric', default=False, action='store_true',
                    help='Apply photometric transform to training images')
parser.add_argument('--mode', default='fast', type=str,
                    choices=['fast', 'cyclegan', 'mixed', 'silhouettes'],
                    help='Which style transfer method to use.')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--cosine', default=False, action='store_true',
                    help='Use cosine similirity instead of MSE for feature consistency.')
parser.add_argument('--preserve_colours', default='nopreserve', type=str,
                    choices=['nopreserve', 'preserve', 'random'],
                    help='When style transferring with fast method, preserve source colour or not, '
                         'or randomly choose per example')
parser.add_argument('--no_prep', default=False, action='store_true',
                    help='Dont prep style transferred dataset.')
parser.add_argument('--random_sample', default=1, type=int,
                    help='Add flag to keep full image during training, rather than randomly sample an example and crop')
parser.add_argument('--expand', default=1, type=int,
                    help='Add flag to keep make sampled image fill entire 300x300 during training, rather than '
                         'resizing and placing randomly')
parser.add_argument('--basic_trans', default=False, action='store_true',
                    help='Use a basic transform of just loading images and randomly flipping and normalizing. '
                         'No randomly samplingor expanding')
parser.add_argument('--generate_ps', default=1, choices=[0, 1], type=int,
                    help='Generate pseudolabels or just existing ones in output_dir/ps_labels.json')
args = parser.parse_args()


def main(args, cfg):
    print(args)

    if args.basic_trans:
        args.expand = False
        args.sample = False

    if args.mode == 'watercolour':
        args.mode = 'watercolor'

    if args.max_its is None:
        if args.train:
            args.max_its = 10000
        elif args.pseudolabel:
            args.max_its = 5000

    # args.max_its += 1000  # add 1000 iterations to run validation over
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
        model_description = str(args.target_domain)
        model_description += '-' + str(args.mode)
        if args.cosine:
            model_description += '-cosine'
        if args.photometric:
            model_description += '-photometric'
        model_description += '-' + args.preserve_colours

        output_dir = os.path.join('outputs', model_description)
    else:
        output_dir = args.output_dir

    print("Saving outputs to {}".format(output_dir))

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
        if args.checkpoint is not None:
            assert os.path.exists(args.checkpoint)
            model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        print("\nTraining Base Model on Stylized Photos pairs....")
        if not args.pseudolabel:
            model = trainer.base_trainer(model, args, output_dir, stylized_root, num_classes)

            # save model.

        print("\nTraining with Joint Datasety of Pseudolabelled Art and Stylized Photos pairs....")
        model = train_pseudolabel.pseudolabel_trainer(model, args, output_dir, stylized_root, num_classes)

    elif args.eval:
        # set up model
        # load weights
        if args.checkpoint is not None:
            assert os.path.exists(args.checkpoint)
            state_dict_to_load = torch.load(args.checkpoint, map_location='cpu')
        else:
            trained_model = "weights/ssd300_mAP_77.43_v2.pth"
            state_dict_to_load = torch.load(trained_model, map_location='cpu')
            output_dir = 'outputs/base-model'
            os.makedirs(output_dir, exist_ok=True)

            # load base model weights
        print('Model loaded.')

        model.load_state_dict(state_dict_to_load)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create datasetr
        dataset_mean = (104, 117, 123)
        set_type = 'test'
        eval_dataset = datasets.ArtDetection(root=args.style_root, transform=BaseTransform(300, dataset_mean),
                                             target_domain=args.target_domain, set_type=set_type,
                                             target_transform=VOCAnnotationTransform())

        # set up dataloaders
        model.eval()
        if torch.cuda.is_available():
            model.cuda()

        accuracy_dict = eval.evaluate(model, eval_dataset, os.path.join(output_dir, 'evaluate'))

        output_file = os.path.join(output_dir, 'accuracy_dict.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(accuracy_dict, f)


if __name__ == '__main__':
    main(args, cfg)
