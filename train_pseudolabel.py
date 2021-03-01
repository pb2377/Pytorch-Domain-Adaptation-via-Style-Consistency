import os
import torch
import eval
import copy
from trainer import boxes_from_dets, visualize_boxes
from torch.utils.tensorboard import SummaryWriter
from cfg import *
from data import datasets, style_detection_collate, BaseTransform, VOCAnnotationTransform
from models.layers.modules import MultiBoxLoss, FeatureConsistency
from utils.augmentations import StyleAugmentation
import torch.utils.data as data
import preprocess
import json
from utils.utils import report_and_save
import pseudolabel
from itertools import cycle


def pseudolabel_trainer(model, args, output_dir, stylized_root, num_classes):
    # check output directory
    if args.checkpoint is not None:
        # find assumed base weights...
        guess_path = os.path.join(output_dir, 'weights', 'ssd300-final.pth')
        print('No checkpoint given, loading checkpoint from {}'.format(guess_path))
        assert os.path.exists(guess_path)
        state_dict_to_load = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict_to_load)

    print('Generating Pseudolabels...')
    dataset_mean = (104, 117, 123)
    pseudo_dataset = datasets.ArtDetection(root=args.style_root, transform=BaseTransform(300, dataset_mean),
                                           target_domain=args.target_domain, set_type='train',
                                           target_transform=VOCAnnotationTransform())
    pslabels = pseudolabel.pseudolabel(model, pseudo_dataset, args.pthresh, overlap_thresh=args.overlap_thresh)

    print("Saving pseudolabels JSON file to {}...".format(os.path.join(output_dir, 'pslabels.json')))
    with open(os.path.join(output_dir, 'pslabels.json'), 'w') as fp:
        json.dump(pslabels, fp)

    # Source, pseudolablled and validation datasets
    sc_loader, ps_loader, val_data = get_dataloaders(args, stylized_root, pslabels)

    # training criterion for source data
    ssd_criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, cfg, torch.cuda.is_available(),
                                 neg_thresh=0.)
    style_criterion = FeatureConsistency(cosine=args.cosine)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # training criterion for pseudolabelled example -- negative threshold adusted
    neg_thresh = args.nthresh
    ps_criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, cfg, torch.cuda.is_available(),
                                neg_thresh=neg_thresh)

    ps_pair = (ps_loader, ps_criterion)  # dataloader and ssd criterion for pseudolabelled image pairs
    sc_pair = (sc_loader, ssd_criterion)  # dataloader and ssd criterion for source image pairs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure datasets are fully preprocessed
    preprocess.check_preprocess(args, sc_loader, stylized_root, pseudolabel=False)
    preprocess.check_preprocess(args, ps_loader, stylized_root, pseudolabel=True)

    # raise NotImplementedError
    args.max_its = 5000
    print("Setting max iterations to 5000 for pseudolabel training.")
    model, best_model, best_map, accuracy_history = train(model, ps_pair, sc_pair, optimizer, val_data, args.max_its,
                                                          output_dir, log_freq=args.log_freq, test_freq=args.test_freq,
                                                          aux_criterion=style_criterion)
    report_and_save(model, best_model, best_map, accuracy_history, output_dir,  args.max_its, pseudolabel=True)
    return model


def get_dataloaders(args, stylized_root, pslabels):
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

    # pseudolabel datasets
    ps_dataset = datasets.PseudolabelDataset(pslabels, args.style_root,
                                             args.target_domain, transform=train_transform,
                                             stylized_root=stylized_root, mode=args.mode)

    ps_loader = data.DataLoader(ps_dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True, collate_fn=style_detection_collate,
                                pin_memory=True)
    return train_loader, ps_loader, val_data


def train(model, ps_pair, sc_pair, optimizer, val_dataset, max_iter, output_path,
                      log_freq=100, test_freq=100, aux_criterion=None):
    """
    As standard trainer but with mixed dataset of Pseudolabels and Labelled VOC data, and train much like "frustratingly
    easy few shot learning" and evaluate on batches of the source and novel (pseudo) examples.

    Also include the addition to only use the labelled examples to generate hard negatives from the synthesized artwork.
    :return:
    """
    ps_loader, criterion_ps = ps_pair  # pseudolabel dataloader and criterion
    sc_loader, criterion_sc = sc_pair  # souerce dataloader and criterion
    del ps_pair, sc_pair
    iteration = 0
    accuracy_history = []
    best_map = 0.0
    best_model = copy.deepcopy(model.state_dict())

    # Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(output_path, 'pseudolabel'))
    writer.add_scalar("Test_mAP", 0., 0)

    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    while iteration <= max_iter:
        for ps_batch, sc_batch in zip(cycle(ps_loader), sc_loader):
            ps_images, ps_style_ims, ps_targets = ps_batch
            sc_images, sc_style_ims, sc_targets = sc_batch

            optimizer.zero_grad()

            if iteration > max_iter:
                break

            # combine into single batch
            images_batch = torch.cat((ps_images, ps_style_ims, sc_images, sc_style_ims), 0)

            if torch.cuda.is_available():
                images_batch = images_batch.cuda()
                ps_targets = [ann.cuda() for ann in ps_targets]
                sc_targets = [ann.cuda() for ann in sc_targets]

            # Pass through model
            out, aux_outputs = model(images_batch)

            # Split outputs back to pseudolabel and source
            ps_out = []
            sc_out = []
            for output in out[:-1]:
                ps_out.append(output[:ps_images.size(0)*2])
                sc_out.append(output[ps_images.size(0)*2:])
            ps_out.append(out[-1])
            sc_out.append(out[-1])

            # Pseudolabel Example Losses
            ps_targets = ps_targets + ps_targets
            loss_l_ps, loss_c_ps = criterion_ps(ps_out, ps_targets)
            loss_ps = loss_l_ps + loss_c_ps
            loss_aux_ps = 0.
            # First half of batch in aux outputs are pseudolabel items
            sz_1 = ps_images.size(0)
            sz_2 = sz_1 * 2
            for idx in range(len(aux_outputs)):
                aux_base = aux_outputs[idx][:sz_1, :, :, :]
                aux_sty = aux_outputs[idx][sz_1:sz_2, :, :, :]
                loss_aux_ps += aux_criterion(aux_base, aux_sty)

            # Source Example Losses
            sc_targets = sc_targets + sc_targets
            loss_l_sc, loss_c_sc = criterion_sc(sc_out, sc_targets)
            loss_sc = loss_l_sc + loss_c_sc

            loss_aux_sc = 0.
            # Second half of batch in aux outputs are source items
            sz_1 = sz_2
            sz_2 = sz_1 + sc_images.size(0)
            for idx in range(len(aux_outputs)):
                # l1, l2 = aux_criterion(aux_outputs_ph[idx], aux_outputs_st[idx])
                aux_base = aux_outputs[idx][sz_1:sz_2, :, :, :]
                aux_sty = aux_outputs[idx][sz_2:, :, :, :]
                loss_aux_sc += aux_criterion(aux_base, aux_sty)
            loss_sc += loss_aux_sc

            # Combine losses
            # print('{:.3f} -- {:.3f} -- {:.3f} -- {:.3f}'.format(loss_ps, loss_aux_ps, loss_sc, loss_aux_sc))
            loss = loss_ps + loss_sc + loss_aux_ps + loss_aux_sc
            loss /= 2
            loss.backward()
            optimizer.step()

            if iteration % log_freq == 0:
                writer.add_scalar("Loss_PS", loss_ps.item(), iteration)
                writer.add_scalar("Conf_loss_PS", loss_c_ps.item(), iteration)
                writer.add_scalar("Loc_loss_PS", loss_l_ps.item(), iteration)
                writer.add_scalar("Style_loss", loss_aux_ps.cpu().numpy().float(), iteration)

            if iteration != 0 and iteration % 5000 == 0:
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

                    writer.add_scalar("Test_mAP", accuracy_history[-1]['mAP'], iteration)
                    # keep best  model
                    if accuracy_history[-1]['mAP'] > best_map:
                        best_map = accuracy_history[-1]['mAP']
                        best_model = copy.deepcopy(model.state_dict())

                    # summary writer
            else:
                if not iteration % 1000 and iteration > 0:
                    model.eval()
                    eval.FULL_REPORT = False
                    acc = eval.evaluate(model, val_dataset, os.path.join(output_path, 'tempdets'.format(iteration)))
                    writer.add_scalar("Test_mAP", acc['mAP'], iteration)
                    model.train()
                    # summary writer

            iteration += 1
    return model, best_model, best_map, accuracy_history