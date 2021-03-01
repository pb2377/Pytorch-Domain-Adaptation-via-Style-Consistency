import os
import argparse
import models
import torch
from data import datasets, style_detection_collate, BaseTransform, VOCAnnotationTransform
from models.layers.modules import MultiBoxLoss, FeatureConsistency
import torch.utils.data as data
import trainer
from utils.augmentations import StyleAugmentation
import numpy as np
import pickle
import json
import preprocess
import glob
import pseudolabel
import eval

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
# add args
parser.add_argument('--train', default=False, action='store_true', help='Flag to train model')
parser.add_argument('--pseudolabel', '--ps', default=False, action='store_true',
                    help='Flag to generate and train with pseudolabel dataset.')
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
parser.add_argument('--log_freq', default=1000, type=int, help='Frequency to report training losses.')
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

        # load checkpoint
        if args.checkpoint is not None:
            if os.path.exists(args.checkpoint):
                state_dict_to_load = torch.load(args.checkpoint, map_location='cpu')
                model.load_state_dict(state_dict_to_load)
                print('Loading model from {}'.format(args.checkpoint))
            else:
                raise OSError('Cannot find checkoint {}'.format(args.checkpoint))

        # set up dataloaders
        train_transform = StyleAugmentation(cfg['min_dim'], MEANS, args.photometric, random_sample=args.random_sample,
                                            expand=args.expand)

        train_dataset = datasets.StylizedVOCDetection(args.voc_root, args.target_domain,
                                                      image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                                                      transform=train_transform, dataset_name='VOC0712',
                                                      stylized_root=stylized_root, mode=args.mode)

        train_loader = data.DataLoader(train_dataset, args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True, collate_fn=style_detection_collate,
                                       pin_memory=True)

        # validation loader
        val_data = datasets.ArtDetection(root=args.style_root, target_domain=args.target_domain, set_type='test',
                                         transform=BaseTransform(300, MEANS))

        # training sub functions
        ssd_criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, cfg, torch.cuda.is_available(),
                                 neg_thresh=0.)
        style_criterion = FeatureConsistency(cosine=args.cosine)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not args.pseudolabel:
            with open(os.path.join(output_dir, 'commandline_args.txt'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

            if not os.path.exists(os.path.join(output_dir, 'weights')):
                os.makedirs(os.path.join(output_dir, 'weights'))

            # Ensure dataset is fully preprocessed
            if args.mode == 'fast':
                if not args.no_prep:
                    preserve_colour = False
                    if args.preserve_colours == 'random':
                        preserve_colour = None
                    elif args.preserve_colours == 'preserve':
                        preserve_colour = True
                    print('Stylizing Dataset...')
                    preprocess.preprocess(train_dataset, args.target_domain, args.max_its, args.batch_size, args.style_root,
                                          stylized_root, set_type='train', preserve_colour=preserve_colour,
                                          pseudo=args.pseudolabel)

            model, best_model, best_map, accuracy_history = trainer.train(model, ssd_criterion, optimizer, train_loader,
                                                                          val_data, args.max_its, output_dir,
                                                                          log_freq=args.log_freq, test_freq=args.test_freq,
                                                                          aux_criterion=style_criterion)
            report_and_save(model, best_model, best_map, accuracy_history, output_dir, pseudolabel=False)

        print('\nBegin Pseudolabel Training...')
        # copy base model to pseudolabel_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.checkpoint is not None:
            # find assumed base weights...
            guess_path = os.path.join(output_dir, 'weights', 'ssd300-final.pth')
            print('No checkpoint given, loading checkpoint from {}'.format(guess_path))
            assert os.path.exists(guess_path)
            state_dict_to_load = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict_to_load)

        if args.generate_ps:
            print('Generating Pseudolabels...')
            dataset_mean = (104, 117, 123)
            pseudo_dataset = datasets.ArtDetection(root=args.style_root, transform=BaseTransform(300, dataset_mean),
                                                   target_domain=args.target_domain, set_type='train',
                                                   target_transform=VOCAnnotationTransform())

            # pseudolabel dataset
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
            pslabels = pseudolabel.pseudolabel(model, pseudo_dataset, args.pthresh, overlap_thresh=args.overlap_thresh)

            print("Saving JSON file...")
            with open(os.path.join(output_dir, 'pslabels.json'), 'w') as fp:
                json.dump(pslabels, fp)
            print("...Complete")
        else:
            # get pre-generated pseudolabels
            pseudolabels_file = os.path.join(output_dir, 'pslabels.json')
            assert os.path.exists(pseudolabels_file)
            print('Loading pseudolabels from {}'.format(pseudolabels_file))
            with open(pseudolabels_file) as json_file:
                pslabels = json.load(json_file)

            if not args.generate_ps:
                n_labels = 0
                for k, v in pslabels.items():
                    n_labels += len(v)
                print("--- {} Detections Pseudolabeled in {} Images ---".format(n_labels, len(pslabels.keys())))

        # pseudolabel datasets
        ps_dataset = datasets.PseudolabelDataset(pslabels, args.style_root,
                                                    args.target_domain, transform=train_transform,
                                                    stylized_root=stylized_root, mode=args.mode)

        ps_loader = data.DataLoader(ps_dataset, args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True, collate_fn=style_detection_collate,
                                       pin_memory=True)

        # train
        neg_thresh = args.nthresh
        ps_criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, cfg, torch.cuda.is_available(),
                                 neg_thresh=neg_thresh)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ensure dataset is fully preprocessed
        if args.mode == 'fast':
            if not args.no_prep:
                preserve_colour = False
                if args.preserve_colours == 'random':
                    preserve_colour = None
                elif args.preserve_colours == 'preserve':
                    preserve_colour = True
                preprocess.preprocess(train_dataset, args.target_domain, args.max_its, args.batch_size,
                                      args.style_root, stylized_root, set_type='train', preserve_colour=preserve_colour,
                                      pseudo=args.pseudolabel)

        # raise NotImplementedError
        args.max_its = 5000
        print("Setting max iterations to 5000 for pseudolabel training.")
        ps_pair = (ps_loader, ps_criterion)  # dataloader and ssd criterion for pseudolabelled image pairs
        sc_pair = (train_loader, ssd_criterion)  # dataloader and ssd criterion for source image pairs
        model, best_model, best_map, accuracy_history = trainer.pseudolabel_train(model, ps_pair, sc_pair, optimizer,
                                                                                  val_data, args.max_its, output_dir,
                                                                                  log_freq=args.log_freq,
                                                                                  test_freq=args.test_freq,
                                                                                  aux_criterion=style_criterion)
        report_and_save(model, best_model, best_map, accuracy_history, output_dir, pseudolabel=True)

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


# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

cfg = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

if __name__ == '__main__':
    main(args, cfg)
