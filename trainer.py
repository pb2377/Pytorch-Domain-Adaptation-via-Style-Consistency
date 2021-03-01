import os
import torch
import eval
import copy
from utils.utils import export_images
from torch.utils.tensorboard import SummaryWriter
from cfg import *
from data import datasets, style_detection_collate, BaseTransform, VOCAnnotationTransform
from models.layers.modules import MultiBoxLoss, FeatureConsistency
from utils.augmentations import StyleAugmentation
import torch.utils.data as data
import preprocess
import json
from utils.utils import report_and_save


def base_trainer(model, args, output_dir, stylized_root, num_classes):
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

        model, best_model, best_map, accuracy_history = train(model, ssd_criterion, optimizer, train_loader,
                                                                      val_data, args.max_its, output_dir,
                                                                      log_freq=args.log_freq, test_freq=args.test_freq,
                                                                      aux_criterion=style_criterion)
        report_and_save(model, best_model, best_map, accuracy_history, output_dir, pseudolabel=False)
        return model


def train(model, criterion, optimizer, train_loader, val_dataset, max_iter, output_path, log_freq=1000, test_freq=100,
          aux_criterion=None):
    iteration = 0
    epoch = 0
    accuracy_history = []
    best_map = 0.0
    best_model = copy.deepcopy(model.state_dict())

    # Tensorboard writer
    writer = SummaryWriter(log_dir=output_path)
    writer.add_scalar("Test_mAP", 0., 0)

    while iteration <= max_iter:
        for images, style_ims, targets in train_loader:
            optimizer.zero_grad()

            if iteration > max_iter:
                break

            if torch.cuda.is_available():
                style_ims = style_ims.cuda()
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]

            # combine image batch
            images = torch.cat((images, style_ims), 0)
            targets = targets * 2

            out, aux_outputs = model(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c

            loss_aux = 0.

            for idx in range(len(aux_outputs)):
                aux_ph = aux_outputs[idx][:images.size(0)//2, :, :, :]
                aux_st = aux_outputs[idx][images.size(0)//2:, :, :, :]
                loss_aux += aux_criterion(aux_ph, aux_st)

            loss = loss + loss_aux

            loss.backward()
            optimizer.step()

            if not iteration % log_freq:
                # make summary writer for loc lss, conf loss, consistency_loss,
                writer.add_scalar("Loss", loss.item(), iteration)
                writer.add_scalar("Conf_loss", loss_c.item(), iteration)
                writer.add_scalar("Loc_loss", loss_l.item(), iteration)
                writer.add_scalar("Style_loss", loss_aux, iteration)

            if iteration > 0 and not iteration % 5000:
                print('Saving state, iter:', iteration)
                save_path = os.path.join(output_path, 'weights', 'iteraton-{}.pth'.format(iteration))
                torch.save(model.state_dict(), save_path)

            if iteration >= max_iter - (10*test_freq):
                if iteration == max_iter or iteration % test_freq == 0:
                    model.eval()
                    eval.FULL_REPORT = False
                    accuracy_history.append(eval.evaluate(model, val_dataset,
                                                          os.path.join(output_path, 'iteration-{}'.format(iteration))))
                    model.train()

                    # keep best  model
                    writer.add_scalar("Test_mAP", accuracy_history[-1]['mAP'], iteration)
                    if accuracy_history[-1]['mAP'] > best_map:
                        best_map = accuracy_history[-1]['mAP']
                        best_model = copy.deepcopy(model.state_dict())

                    # Add to summary writer
            else:
                if not iteration % 1000 and iteration > 0:
                    model.eval()
                    eval.FULL_REPORT = False
                    acc = eval.evaluate(model, val_dataset, os.path.join(output_path, 'tempdets'.format(iteration)))
                    writer.add_scalar("Test_mAP", acc['mAP'], iteration)
                    model.train()

            iteration += 1
        epoch += 1

    return model, best_model, best_map, accuracy_history


def boxes_from_dets(detections, thresh=0.25):
    targets = []
    confs = []
    for bidx in range(detections.size(0)):
        pthresh = thresh
        while True:
            bidx_targets = None
            bidx_scores = None
            for cidx in range(1, detections.size(1)):
                    scores = detections[bidx, cidx, :, 0].squeeze()
                    bboxes = detections[bidx, cidx, :, 1:]

                    # scores = scores.view(-1, 1)
                    lookup = scores > pthresh
                    if lookup.sum() > 0:
                        scores = scores[lookup]
                        bboxes = bboxes[lookup, :]
                        lbls = torch.ones(bboxes.size(0), 1) * (cidx - 1)
                        bboxes = torch.cat((bboxes, lbls), 1)
                        if bidx_targets is None:
                            bidx_targets = bboxes
                            bidx_scores = scores
                        else:
                            bidx_targets = torch.cat((bidx_targets, bboxes), 0)
                            bidx_scores = torch.cat((bidx_scores, scores), 0)

            if bidx_targets is None:
                pthresh *= 0.9
            else:
                break
        targets.append(bidx_targets)
        confs.append(bidx_scores)
    return confs, targets


def visualize_boxes(images, style_ims, targets, image_base=None):
    count = 0
    if image_base is None:
        image_base = 'test_imgs'

    for bidx in range(images.size(0)):

        image_out = os.path.join(image_base, 'example_{}.jpg'.format(count))
        count += 1

        if not os.path.exists(os.path.dirname(image_out)):
            os.makedirs(os.path.dirname(image_out))

        bidx_anno = targets[bidx].tolist()
        annos = []
        for anno in bidx_anno:
            lbl = int(anno[-1])
            anno = [i*images.size(-1) for i in anno[:-1]]
            anno.append(lbl)
            annos.append(anno)
        if style_ims is None:
            export_images([images[bidx, :, :, :]], image_out, annos=annos)
        else:
            export_images([images[bidx, :, :, :], style_ims[bidx, :, :, :]], image_out,
                          annos=annos)
    pass
